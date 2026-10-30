"""
Voltics 배터리 엔진 — 배터리 수명 감가를 고려한 입찰·운영 계산

구성
  1) 배터리 감가 모델 (비선형): 기준 감가 × SoC 구간 계수 × C-rate 계수 × 방전깊이(DoD) 계수
     - 배터리 종류(LFP/NMC)마다 계수가 다름 (CHEMISTRY)
     - SoC 하드 리밋 5~95%: 사용자가 어떻게 설정해도 이 밖으로는 운전하지 않음
  2) 시장 규칙 (MARKET_RULES): 제주 시범사업(별표 33) / 육지 확대 시나리오 — 규칙만 바꿔 끼우면 됨
  3) 내일 입찰: bid_curve (가격-수량 3구간), plan_day (수량 계획, 비교용)
  4) 당일 판단: decide_hour (충전·방전·대기)
  5) 정산·검증: settle_actual, settle_curve, backtest_year

※ 감가 계수는 일반적인 경향을 반영한 가정값이에요. 실제 서비스에서는 제조사 수명 데이터로 맞춥니다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 설정값 (화면에서 슬라이더로 바꿀 수 있음)
# ---------------------------------------------------------------------------
@dataclass
class BatteryConfig:
    energy_kwh: float | None = None  # 사업자가 입력한 배터리 정격 용량(kWh). None이면 hours로 계산
    power_kw: float | None = None    # 사업자가 입력한 최대 충방전 출력(kW). None이면 c_rate로 계산
    soh: float = 1.0                 # 현재 건강도(State of Health). 0.85면 정격의 85%만 쓸 수 있음
    hours: float = 1.0               # 배터리 크기 = 발전소 용량 × 몇 시간 분량 (1.0 → 1MW 발전소에 1MWh)
    c_rate: float = 0.5              # 한 시간에 낼 수 있는 출력 = 배터리 용량 × c_rate
    efficiency: float = 0.90         # 충전→방전 왕복 효율
    soc_min: float = 0.10            # 최저 충전량 (물리적 한계)
    soc_max: float = 0.90            # 최고 충전량 (물리적 한계)
    soc_init: float = 0.50           # 시작 충전량
    capex_won_per_kwh: float = 400_000   # 배터리 설치비 (원/kWh)
    cycle_life: float = 6_000        # 수명 사이클 (80% DoD 기준, 이 사이클을 쓰면 용량이 80%로 떨어짐)
    stress_low: float = 0.20         # 이 아래로 내려가면 열화가 빨라지는 구간
    stress_high: float = 0.80        # 이 위로 올라가면 열화가 빨라지는 구간
    stress_factor: float = 2.0       # (옛 시뮬레이터용) 위 구간에서 감가비용 배수
    # --- 비선형 감가 곡선 (배터리 종류별 기본값은 CHEMISTRY) ---
    chem: str = "LFP"
    soc_lo_alpha: float = 1.0        # SoC 20% 아래로 내려갈수록 감가 증가, 0%에서 (1+이 값)배
    soc_hi_alpha: float = 0.5        # SoC 80% 위로 올라갈수록 감가 증가, 100%에서 (1+이 값)배
    dod_k: float = 1.1               # 수명 ∝ DoD^(-k). k>1이면 깊게 쓸수록 kWh당 감가가 커짐
    crate_beta: float = 0.3          # C-rate가 0.5C를 넘을 때 1C당 추가 배수

    def __post_init__(self):
        # SoC 하드 리밋: 어떤 입력이 와도 5~95% 밖으로는 운전하지 않음 (과충전·완전방전 방지)
        self.soc_min = float(min(max(self.soc_min, HARD_SOC_MIN), HARD_SOC_MAX - 0.05))
        self.soc_max = float(max(min(self.soc_max, HARD_SOC_MAX), self.soc_min + 0.05))


HARD_SOC_MIN, HARD_SOC_MAX = 0.05, 0.95

# 배터리 종류별 감가 곡선 기본값 (일반적 경향 기반 가정값 — 제조사 데이터로 교체 가능)
#   LFP: 깊은 방전·고충전에 비교적 강함 / NMC: 고충전(80%↑)과 빠른 충방전에 더 민감
CHEMISTRY = {
    "LFP": dict(soc_lo_alpha=1.0, soc_hi_alpha=0.5, dod_k=1.1, crate_beta=0.3),
    "NMC": dict(soc_lo_alpha=1.0, soc_hi_alpha=1.5, dod_k=1.5, crate_beta=0.6),
}


@dataclass
class MarketConfig:
    bid_factor: float = 1.0          # 입찰량 = 예측 × bid_factor  (모델이 흐린 날 과대예측 → 1보다 작게 하면 유리할 수 있음)
    tolerance_pct: float = 8.0       # 설비용량 대비 이 % 이내 오차는 페널티 없음
    penalty_ratio: float = 1.0       # 허용범위를 넘는 오차 1kWh당 페널티 = SMP × penalty_ratio
    min_output_ratio: float = 0.10   # 별표33: 발전량이 설비용량의 이 비율 미만인 시간은 페널티 면제


# ---------------------------------------------------------------------------
# 시장 규칙 (모듈): 제도가 바뀌거나 육지로 확대되면 여기에 규칙 세트를 추가·수정
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketRules:
    name: str
    tolerance_pct: float          # 임밸런스 허용오차 (설비용량 대비 %) — 현재 기준
    tolerance_2025: float         # 2025년 기준 (과거 재현·1년 성과 계산용)
    min_output_ratio: float       # 발전량이 설비용량의 이 비율 미만이면 페널티 면제
    shortfall_penalty: bool       # 모자람에도 페널티를 매기는지 (별표33: 아니오, 실시간 가격으로 정산)
    max_segments: int             # 시간당 가격-수량 입찰 구간 수 상한
    source: str


MARKET_RULES = {
    "제주": MarketRules("제주 시범사업 (전력시장운영규칙 별표 33)", 6.0, 8.0, 0.10, False, 10,
                       "2026년 4월부터 6%(급전지시 시 1.5%) — 파이낸셜뉴스 2026.9.25 / "
                       "별표 33: 2024년까지 12%, 2025년 8%, 10% 미만 면제"),
    "육지": MarketRules("육지 확대 시나리오 (가정: 제주 규칙 준용)", 6.0, 8.0, 0.10, False, 10,
                       "육지 입찰제도 정산 규칙이 확정되면 이 값을 바꾸세요"),
}


def market_config(market: str, tolerance_pct: float | None = None) -> "MarketConfig":
    r = MARKET_RULES.get(market, MARKET_RULES["제주"])
    return MarketConfig(tolerance_pct=r.tolerance_pct if tolerance_pct is None else tolerance_pct,
                        min_output_ratio=r.min_output_ratio)


# ---------------------------------------------------------------------------
# 비용 계산식
# ---------------------------------------------------------------------------
def base_degradation_cost(b: BatteryConfig) -> float:
    """배터리로 1kWh를 움직일 때(충전 또는 방전) 닳는 비용(원/kWh).
    설치비 ÷ (평생 충방전 가능한 총량). 1사이클 = 충전 1번 + 방전 1번."""
    dod = 0.8
    return b.capex_won_per_kwh / (2 * b.cycle_life * dod)


def rated_energy(b: BatteryConfig, cap_kw: float) -> float:
    return b.energy_kwh if b.energy_kwh is not None else cap_kw * b.hours


def soc_factor(soc, b: BatteryConfig):
    """SoC 구간 계수: 20~80%는 1배, 그 밖으로 갈수록 선형 증가"""
    s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
    f = 1.0 + b.soc_lo_alpha * np.clip((0.2 - s) / 0.2, 0, 1) + b.soc_hi_alpha * np.clip((s - 0.8) / 0.2, 0, 1)
    return f


def crate_factor(kwh_per_hour, b: BatteryConfig, cap_kw: float):
    """C-rate 계수: 1시간에 움직인 양 ÷ 정격용량. 0.5C까지는 1배"""
    Er = rated_energy(b, cap_kw)
    C = np.asarray(kwh_per_hour, dtype=float) / Er if Er > 0 else 0.0
    return 1.0 + b.crate_beta * np.maximum(C - 0.5, 0.0)


def dod_factor(depth, b: BatteryConfig):
    """방전깊이 계수: 80% 깊이를 기준(1배)으로, 수명 ∝ DoD^(-k) 관계에서 나오는 kWh당 감가 비율"""
    d = np.clip(np.asarray(depth, dtype=float), 0.05, 1.0)
    return (d / 0.8) ** (b.dod_k - 1.0)


def plan_deg_cost(b: BatteryConfig, cap_kw: float, lo: float, hi: float, kwh_per_hour: float) -> float:
    """계획·판단용 kWh당 감가: lo~hi 구간에서 kwh_per_hour 속도로 돌린다고 가정한 평균값"""
    c0 = base_degradation_cost(b)
    if hi <= lo:
        return float(c0 * soc_factor(lo, b) * crate_factor(kwh_per_hour, b, cap_kw))
    avg_soc = float(np.mean(soc_factor(np.linspace(lo, hi, 21), b)))
    return float(c0 * avg_soc * crate_factor(kwh_per_hour, b, cap_kw) * dod_factor(hi - lo, b))


def effective_deg_cost(b: BatteryConfig, cap_kw: float, reserve: float = 0.0) -> float:
    """화면 표시용 대표 감가 (최대 출력으로, 예비분 위 구간을 쓴다고 가정)"""
    E = usable_energy(b, cap_kw)
    P = b.power_kw if b.power_kw is not None else E * b.c_rate
    return plan_deg_cost(b, cap_kw, min(b.soc_min + reserve, b.soc_max), b.soc_max, P)


def actual_deg_cost(b: BatteryConfig, cap_kw: float, charge, discharge, soc_path, soc_start) -> float:
    """정산용 실제 감가: 시간마다 (기준 × SoC 계수 × C-rate 계수), 하루 전체에 DoD 계수"""
    c0 = base_degradation_cost(b)
    c, d = np.asarray(charge, float), np.asarray(discharge, float)
    socs = np.asarray(soc_path, float)
    prev = np.r_[soc_start, socs[:-1]]
    mid = (prev + socs) / 2
    hourly = (c + d) * c0 * soc_factor(mid, b) * crate_factor(c + d, b, cap_kw)
    if hourly.sum() <= 0:
        return 0.0
    depth = max(np.max(np.r_[soc_start, socs]) - np.min(np.r_[soc_start, socs]), 0.05)
    return float(hourly.sum() * dod_factor(depth, b))


def usable_energy(b: BatteryConfig, cap_kw: float) -> float:
    rated = b.energy_kwh if b.energy_kwh is not None else cap_kw * b.hours
    return rated * b.soh


# 배터리 예시값 (실제 제품 사양이 아니라 일반적인 범위를 담은 시작값 — 사업자가 자기 값으로 바꿔 쓴다)
PRESETS = {
    "직접 입력": None,
    "LFP 표준형 (예시)":   dict(duration_h=2.0, efficiency=0.88, cycle_life=6000, capex=300_000, soh=1.00, chem="LFP"),
    "NMC 고출력형 (예시)": dict(duration_h=1.0, efficiency=0.90, cycle_life=4000, capex=350_000, soh=1.00, chem="NMC"),
    "사용 중 배터리 (예시)": dict(duration_h=2.0, efficiency=0.85, cycle_life=6000, capex=300_000, soh=0.85, chem="LFP"),
}


# ---------------------------------------------------------------------------
# 노트북 결과(test_predictions.csv, 45만 줄) → 지역별 시간 합계(약 4만 줄)로 줄이기
# ---------------------------------------------------------------------------
def shrink_test_predictions(src_csv, dst_csv):
    t = pd.read_csv(src_csv, encoding="utf-8-sig", parse_dates=["datetime"])
    g = (
        t.groupby(["site", "datetime"], as_index=False)
        .agg(cap_kw=("cap_kw", "sum"), gen_kwh=("gen_kwh", "sum"),
             pred_kwh=("pred_B_예보대응", "sum"), smp=("smp", "first"))
    )
    g["smp"] = g.groupby("site")["smp"].transform(lambda s: s.ffill().bfill())
    g[["cap_kw", "gen_kwh", "pred_kwh", "smp"]] = g[["cap_kw", "gen_kwh", "pred_kwh", "smp"]].round(2)
    g.to_csv(dst_csv, index=False, encoding="utf-8-sig")
    return g


# ---------------------------------------------------------------------------
# 실시간 운영 판단: "이 시간에 배터리를 충전할까, 방전할까, 가만히 둘까?"
#   - 약속량(하루 전 낙찰량)과 예상 실제 발전량의 차이를 본다
#   - 모자란 만큼은 실시간 가격으로 정산(=사 오는 셈), 남는 만큼은 실시간 가격으로 받는다
#   - 배터리 전기는 '지금' 쓰는 것과 '오늘 남은 시간 중 가장 비쌀 때' 쓰는 것을 비교한다
#   - 모든 선택지에서 배터리가 닳는 비용(감가)을 뺀다
# ---------------------------------------------------------------------------
def decide_hour(promised_kwh, expected_kwh, price_now, future_prices, cap_kw,
                b: BatteryConfig, soc_now: float, m: MarketConfig, reserve: float = 0.0,
                export_cap: float | None = None):
    """이번 시간에 충전·방전·대기 중 무엇이 가장 이득인지.

    배터리 속 전기 1kWh의 가치 = 나중에 꺼내 팔 때 받을 돈 − 그때 닳는 비용
                               = max(오늘 남은 시간 최고가 − 감가, 0)
      → 지금 방전하면 이 가치를 지금 써 버리는 것, 지금 충전하면 이 가치를 새로 만드는 것.
    방전은 두 가지:
      - 모자란 만큼 메우기: 최저 충전량까지 써도 됨
      - 약속 안 한 추가 판매: 페널티 없는 범위(허용범위) 안에서만, 예비 충전량은 남겨 두고
    """
    E = usable_energy(b, cap_kw)
    P = b.power_kw if b.power_kw is not None else E * b.c_rate
    eta = np.sqrt(b.efficiency)
    allowed = cap_kw * m.tolerance_pct / 100
    future_best = max(future_prices) if len(future_prices) else 0.0
    c_later = plan_deg_cost(b, cap_kw, min(b.soc_min + reserve, b.soc_max), b.soc_max, P)
    c_deg = c_later                                       # 화면 표시용 대표값
    hold_value = max(future_best - c_later, 0.0)          # 배터리에 남겨 둔 전기 1kWh의 가치 (나중에 꺼낼 때 감가 반영)
    depth = max(b.soc_max - b.soc_min - reserve, 0.05)

    def c_now(kwh, soc_after):
        """지금 이 양을 움직일 때의 kWh당 감가: SoC 구간(지금~움직인 뒤) × C-rate × 평소 운전 깊이"""
        if kwh <= 0 or E <= 0:
            return 0.0
        mid = (soc_now + soc_after) / 2
        return float(base_degradation_cost(b) * soc_factor(mid, b) * crate_factor(kwh, b, cap_kw) * dod_factor(depth, b))

    dev = expected_kwh - promised_kwh                     # + 남음 / - 모자람
    shortfall = max(-dev, 0.0)
    can_dis = max(min(P, (soc_now - b.soc_min) * E * eta), 0.0)              # 최저 충전량까지
    can_dis_extra = max(min(P, (soc_now - b.soc_min - reserve) * E * eta), 0.0)  # 예비 충전량은 남기고
    can_ch = max(min(P, (b.soc_max - soc_now) * E / eta), 0.0)

    def settle_parts(delivered):
        """약속 대비 정산(이번 시간): 약속분 수익은 이미 확정이므로 '차이'에 대한 돈만 계산.
        출력제어 중이면 허용 출력(export_cap)을 넘는 전기는 못 보내고 버려짐."""
        if export_cap is not None:
            delivered = min(delivered, export_cap)
        d = delivered - promised_kwh
        money = d * price_now                                   # 모자라면 -(사 오는 돈), 남으면 +(추가 판매)
        pen = float(penalty_over(d, cap_kw, price_now, m, delivered))   # 별표33: 허용범위 넘게 더 보내면 페널티(10% 미만 면제)
        return money - pen, pen

    def settle(delivered):
        return settle_parts(delivered)[0]

    options = [dict(action="대기", kwh=0.0, now=settle(expected_kwh), pen=settle_parts(expected_kwh)[1],
                    later=0.0, deg=0.0, c_kwh=0.0,
                    why=("배터리를 쓰지 않아요. 출력제어로 못 보내는 전기는 버려져요." if export_cap is not None
                         else "배터리를 쓰지 않고 차이는 실시간 가격으로 정산해요."))]

    # 방전 후보: ① 모자란 만큼 메우기  ② 메우고 + 약속 안 한 추가 판매(허용범위 안, 예비 충전량 제외)
    if export_cap is not None:
        can_dis = can_dis_extra = 0.0                     # 출력제어 중엔 더 보낼 수 없으니 방전 의미 없음
    cover = min(can_dis, shortfall)
    extra = max(min(allowed, can_dis_extra - cover, P - cover), 0.0)
    best_d = None
    for x, kind in [(cover, "메우기"), (cover + extra, "추가 판매")]:
        if x <= 0:
            continue
        cd = c_now(x, soc_now - x / eta / E if E > 0 else soc_now)
        o = dict(action="방전", kwh=x, now=settle(expected_kwh + x), pen=settle_parts(expected_kwh + x)[1],
                 later=-x * hold_value, deg=-x * cd, c_kwh=cd)
        o["total"] = o["now"] + o["later"] + o["deg"]
        sold = f"모자란 {min(x, shortfall):,.0f} kWh를 메우" if shortfall > 0 else ""
        add = x - min(x, shortfall)
        if add > 0:
            sold += (" 고, " if sold else "") + f"약속 안 한 {add:,.0f} kWh를 더 팔아요(페널티 없는 범위 안, 예비 충전량은 남김)"
        else:
            sold += "요"
        o["why"] = f"{sold}. 지금 {price_now:,.0f}원이 앞으로 최고가 {future_best:,.0f}원보다 낫다고 봐요."
        if best_d is None or o["total"] > best_d["total"]:
            best_d = o
    if best_d:
        options.append(best_d)

    # 충전 후보: 남는 만큼만 / 넣을 수 있는 만큼 최대
    if can_ch > 0 and expected_kwh > 0:
        best_c = None
        for y in sorted({min(can_ch, max(dev, 0.0)), min(can_ch, expected_kwh)}):
            if y <= 0:
                continue
            out = y * eta * eta
            cc = c_now(y, soc_now + y * eta / E if E > 0 else soc_now)
            o = dict(action="충전", kwh=y, now=settle(expected_kwh - y), pen=settle_parts(expected_kwh - y)[1],
                     later=out * hold_value, deg=-y * cc, c_kwh=cc,
                     why=f"지금 {price_now:,.0f}원에 팔지 않고 담아 뒀다가 앞으로 최고가 {future_best:,.0f}원일 때 팔아요 "
                         f"(효율 손실 {1 - b.efficiency:.0%}, 넣을 때·뺄 때 모두 감가 반영).")
            o["total"] = o["now"] + o["later"] + o["deg"]
            if best_c is None or o["total"] > best_c["total"]:
                best_c = o
        if best_c:
            options.append(best_c)

    table = pd.DataFrame(options)
    table["total"] = table["now"] + table["later"] + table["deg"]
    best = table.loc[table["total"].idxmax()]
    eta2 = b.efficiency
    if best.action == "충전":
        cc = best.c_kwh
        lost = 0.0 if (export_cap is not None and expected_kwh > (export_cap or 0)) else price_now
        why_now = dict(a=("지금 팔면 kWh당" if lost else "지금 못 보내서 버리면 kWh당", lost),
                       b=("담았다 나중에 팔면 kWh당", eta2 * hold_value - cc))
    elif best.action == "방전":
        why_now = dict(a=("지금 꺼내 쓰면 kWh당", price_now - best.c_kwh), b=("아껴 뒀다 나중에 쓰면 kWh당", hold_value))
    else:
        alt_c = table[table.action == "충전"]
        alt_d = table[table.action == "방전"]
        if len(alt_d):
            why_now = dict(a=("지금 꺼내 쓰면 kWh당", price_now - float(alt_d.c_kwh.iloc[0])),
                           b=("아껴 뒀다 나중에 쓰면 kWh당", hold_value))
        elif len(alt_c):
            why_now = dict(a=("지금 팔면 kWh당", price_now),
                           b=("담았다 나중에 팔면 kWh당", eta2 * hold_value - float(alt_c.c_kwh.iloc[0])))
        else:
            why_now = None
    info = dict(why_now=why_now, dev=dev, c_deg=c_deg, future_best=future_best, hold_value=hold_value, allowed=allowed,
                can_dis=can_dis, can_dis_extra=can_dis_extra, can_ch=can_ch, reserve=reserve)
    return best, table, info


# ---------------------------------------------------------------------------
# EPSIS(전력통계정보시스템) '시간별 SMP' CSV 읽기 → 날짜별 24시간 가격표
# ---------------------------------------------------------------------------
def read_epsis_hourly(file):
    import io
    import re
    raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
    for enc in ["cp949", "utf-8-sig", "utf-8", "euc-kr"]:
        try:
            t = pd.read_csv(io.BytesIO(raw), encoding=enc)
            break
        except Exception:
            continue
    t.columns = [str(c).strip() for c in t.columns]
    hcols = [c for c in t.columns if re.fullmatch(r"0?([1-9]|1\d|2[0-4])\s*(시|h)?", c)]
    dcol = t.columns[0]
    t = t[[dcol] + hcols].copy()
    t[dcol] = pd.to_datetime(t[dcol].astype(str).str.replace(r"[^0-9]", "", regex=True).str[:8],
                             format="%Y%m%d", errors="coerce")
    t = t.dropna(subset=[dcol])
    long = t.melt(id_vars=dcol, var_name="h", value_name="price")
    long["hour"] = long["h"].str.extract(r"(\d+)")[0].astype(int) - 1      # '1시' = 00~01시 구간
    long["price"] = pd.to_numeric(long["price"].astype(str).str.replace(",", ""), errors="coerce")
    long = long.rename(columns={dcol: "date"})[["date", "hour", "price"]]
    # 아직 오지 않은 날(빈칸이거나 하루 종일 0)은 빼기
    ok = long.groupby("date")["price"].transform(lambda x: x.notna().any() and (x.fillna(0) != 0).any())
    long = long[ok]
    return long.sort_values(["date", "hour"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 내일 입찰 계획: AI 발전량 예측 + 예상 가격 + 배터리 → 시간별 추천 입찰량
#   목표: (각 시간 입찰량 × 예상 가격)의 합 − 배터리 감가비용 을 최대로
#   방법: 선형계획법(LP)으로 24시간 충전·방전량을 한꺼번에 정한다
#   규칙: - 배터리는 내 발전소 전기로만 충전 (계통에서 사 와서 충전하지 않음)
#         - 하루가 끝날 때 배터리 충전량은 시작보다 줄지 않게 (공짜로 빼먹지 않기)
#         - 예비 충전량(reserve)은 계획에 쓰지 않고 남겨 둠 → 예측이 틀릴 때 당일 메우는 용도
# ---------------------------------------------------------------------------
def plan_day(gen_kwh, price, b: BatteryConfig, cap_kw, soc_start=0.5, reserve=0.0):
    from scipy.optimize import linprog

    g = np.asarray(gen_kwh, dtype=float)
    p = np.asarray(price, dtype=float)
    n = len(g)
    E = usable_energy(b, cap_kw)
    P = b.power_kw if b.power_kw is not None else E * b.c_rate
    eta = np.sqrt(b.efficiency)
    soc0 = min(max(soc_start, 0.0), 1.0)
    # 하한 = 최저 충전량 + 예비분. 단, 지금 충전량이 그보다 낮으면 '지금보다 더 쓰지 않기'까지만 요구
    lo = min(b.soc_min + reserve, b.soc_max, soc0)
    hi = max(b.soc_max, soc0)
    c_deg = plan_deg_cost(b, cap_kw, lo, hi, P)     # 이 구간·이 속도로 돌릴 때의 kWh당 평균 감가 (비선형 반영)

    if E <= 0 or P <= 0:
        c = d = np.zeros(n)
    else:
        # 변수: [c_0..c_{n-1}, d_0..d_{n-1}]  (c=충전량, d=방전량, kWh)
        # 최대화: Σ p(-c + d) − c_deg Σ(c + d)   → linprog는 최소화라서 부호 반대로
        cost = np.concatenate([p + c_deg, -p + c_deg])
        # SOC_t = soc0 + (η Σ_{k≤t} c_k − Σ_{k≤t} d_k / η) / E  ∈ [lo, hi]
        L = np.tril(np.ones((n, n)))
        A_soc = np.hstack([L * eta, -L / eta]) / E
        A_ub = np.vstack([A_soc, -A_soc, -A_soc[-1:]])
        b_ub = np.concatenate([np.full(n, hi - soc0), np.full(n, soc0 - lo), [0.0]])   # 마지막 줄: 끝 SOC ≥ 시작
        bounds = [(0, min(P, gi)) for gi in g] + [(0, P)] * n
        r = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        x = r.x if r.success else np.zeros(2 * n)
        c, d = x[:n], x[n:]
        c[c < 1e-6] = 0
        d[d < 1e-6] = 0

    soc = soc0 + np.cumsum(c * eta - d / eta) / E if E > 0 else np.full(n, soc0)
    bid = g - c + d
    out = pd.DataFrame({"gen_kwh": g, "price": p, "charge_kwh": c, "discharge_kwh": d, "bid_kwh": bid, "soc": soc})
    summary = {
        "배터리 없이 예상 수입(원)": float((g * p).sum()),
        "배터리 포함 예상 수입(원)": float((bid * p).sum()),
        "배터리 감가비용(원)": float(((c + d) * c_deg).sum()),
    }
    summary["순이득(원)"] = summary["배터리 포함 예상 수입(원)"] - summary["배터리 감가비용(원)"] - summary["배터리 없이 예상 수입(원)"]
    return out, summary


def expected_prices(prices_long: pd.DataFrame, days: int = 7) -> pd.Series:
    """최근 며칠의 시간대별 평균 가격 → '내일 예상 가격' (0~23시)"""
    recent = prices_long[prices_long["date"] >= prices_long["date"].max() - pd.Timedelta(days=days - 1)]
    return recent.groupby("hour")["price"].mean().reindex(range(24)).ffill().bfill()


# ---------------------------------------------------------------------------
# 유사일 기반 가격 예측: "내일과 날씨(햇빛 양)가 비슷했던 과거 날들"의 시간대별 가격 평균
#   - 제주 한낮 가격은 맑아서 태양광이 넘치면 0원까지 떨어진다 → 단순 평균은 맑은 날·흐린 날을 섞어 버림
#   - 햇빛 양의 지표 = 그날 태양광 이용률(발전량 ÷ 용량 ÷ 24)
#       과거: 실측 이용률 (site_hourly_2025.csv)  /  내일: AI 예측 이용률
#   - 같은 계절(±1개월)에서 이용률이 가장 비슷한 k일을 골라 평균
#   - 과거 가격과 과거 발전량이 겹치는 날이 부족하면 최근 7일 평균으로 대체
# ---------------------------------------------------------------------------
def similar_day_prices(prices_long, daily_cf, target_cf, target_date, k=10, min_overlap=20):
    mat = prices_long.pivot_table(index="date", columns="hour", values="price").reindex(columns=range(24))
    mat = mat.dropna(thresh=20).ffill(axis=1).bfill(axis=1)
    cf = daily_cf.copy()
    cf.index = pd.to_datetime(cf.index)
    common = mat.index.intersection(cf.index)
    if len(common) < min_overlap:
        return expected_prices(prices_long), dict(method="recent", n=0)

    td = pd.Timestamp(target_date)
    months = {((td.month - 1 + d) % 12) + 1 for d in (-1, 0, 1)}
    pool = [d for d in common if d.month in months]
    if len(pool) < k:
        pool = list(common)
    diff = (cf.loc[pool] - target_cf).abs().sort_values()
    days = diff.index[:k]
    est = mat.loc[days].mean()
    return est, dict(method="similar", n=len(days), days=list(days),
                     cf_mean=float(cf.loc[days].mean()), target_cf=float(target_cf))


def daily_capacity_factor(hist: pd.DataFrame, sites) -> pd.Series:
    """과거 실측 → 날짜별 이용률 (여러 지역이면 평균)"""
    h = hist[hist["site"].isin(sites)].copy()
    h["date"] = h["datetime"].dt.normalize()
    d = h.groupby(["site", "date"]).agg(gen=("gen_kwh", "sum"), cap=("cap_kw", "first"))
    d["cf"] = d["gen"] / (d["cap"] * 24)
    return d.groupby("date")["cf"].mean()


# ---------------------------------------------------------------------------
# 유사일 가격 예측 (ASOS 버전): 2023~어제 전체에서 '내일과 맑은 정도가 비슷한 날'을 찾는다
#   1) 내일이 '이 계절 기준으로 얼마나 맑은 날인지'를 백분위로 나타낸다
#        (AI 예측 이용률이 같은 계절 과거 실측 이용률 중 몇 % 위치인지)
#   2) 과거 날짜들도 ASOS 일사량으로 같은 계절 안에서의 백분위를 구한다
#   3) 백분위가 가장 가까운 k일을 고른다 (같으면 최근 날짜 우선)
#   4) '모양은 과거, 수준은 최근': 고른 날의 가격을 그 무렵 평균으로 나눠 모양만 가져오고,
#      최근 28일 평균 가격을 곱한다 → 연료비 변화로 가격 수준이 달라져도 맞춰짐
# ---------------------------------------------------------------------------
def _doy_window(dates, target, window):
    doy = pd.DatetimeIndex(dates).dayofyear.values
    t = pd.Timestamp(target).dayofyear
    diff = np.abs(doy - t)
    return np.minimum(diff, 366 - diff) <= window


def season_percentile(value, history: pd.Series, target_date, window=30):
    h = history.dropna()
    h = h[_doy_window(h.index, target_date, window)]
    if len(h) == 0:
        return 0.5
    return float((h < value).mean() + 0.5 * (h == value).mean())


def similar_day_prices_asos(prices_long, sun_daily: pd.Series, target_pct, target_date,
                            k=10, window=30, recency_weight=0.15):
    mat = prices_long.pivot_table(index="date", columns="hour", values="price").reindex(columns=range(24))
    mat = mat.dropna(thresh=20).ffill(axis=1).bfill(axis=1)
    sun = sun_daily.dropna()
    sun.index = pd.to_datetime(sun.index)

    # 그 날 무렵(앞뒤 14일) 평균 가격 → '모양' 계산용
    daily_mean = mat.mean(axis=1)
    level_around = daily_mean.rolling(29, center=True, min_periods=7).mean()

    common = mat.index.intersection(sun.index)
    pool = common[_doy_window(common, target_date, window)]
    if len(pool) < k:
        return None, dict(method="none")

    # 각 날짜의 '같은 계절 안에서의 맑음 백분위'
    pct = pd.Series({d: season_percentile(sun[d], sun, d, window) for d in pool})
    years_ago = (pd.Timestamp(target_date) - pool.to_series()).dt.days / 365
    score = (pct - target_pct).abs() + recency_weight * years_ago.values
    days = score.sort_values().index[:k]

    shape = mat.loc[days].div(level_around.loc[days].replace(0, np.nan), axis=0).mean()
    recent_level = daily_mean[daily_mean.index > daily_mean.index.max() - pd.Timedelta(days=28)].mean()
    est = (shape * recent_level).fillna(recent_level)
    return est, dict(method="asos", n=len(days), days=list(days), target_pct=target_pct,
                     pct_mean=float(pct.loc[days].mean()), recent_level=float(recent_level))


# ---------------------------------------------------------------------------
# 과거 날짜 재현: "그날 Voltics를 썼다면?"
#   - 전날 기준으로 알 수 있었던 정보만 사용 (그날 이후 데이터는 안 봄)
#       AI 예측 발전량 = 그날의 모델 B 예측값, 예상 가격 = 그 전날까지의 데이터로 유사일 예측
#   - 실제 결과 = 그날의 실제 발전량 + 실제 가격으로 정산
#   - 비교: 배터리 없이 AI 예측대로 입찰 vs Voltics 입찰표(배터리 포함)
# ---------------------------------------------------------------------------
def settle_actual(bid, actual_gen, charge, discharge, price, cap_kw, b: BatteryConfig, m: MarketConfig,
                  soc_start: float = 0.5):
    """계획대로 배터리를 돌렸을 때의 실제 정산. 실제 배터리 상태(SOC)를 시간마다 따라간다.
    - 충전: 계획량, 그 시간 실제 발전량, 남은 빈자리 중 작은 만큼만
    - 방전: 계획량, 배터리에 실제로 들어 있는 양 중 작은 만큼만 (흐려서 못 채웠으면 못 꺼냄)"""
    E = usable_energy(b, cap_kw)
    eta = np.sqrt(b.efficiency)
    n = len(actual_gen)
    soc = min(max(soc_start, 0.0), 1.0)
    c = np.zeros(n)
    d = np.zeros(n)
    socs = np.zeros(n)
    for t in range(n):
        if E > 0:
            room = max((b.soc_max - soc) * E / eta, 0.0)
            c[t] = min(charge[t], actual_gen[t], room)
            soc += c[t] * eta / E
            avail = max((soc - b.soc_min) * E * eta, 0.0)
            d[t] = min(discharge[t], avail)
            soc -= d[t] / eta / E
        socs[t] = soc
    delivered = actual_gen - c + d
    revenue = delivered * price
    penalty = penalty_over(delivered - bid, cap_kw, price, m, delivered)
    deg_total = actual_deg_cost(b, cap_kw, c, d, socs, soc_start)
    return dict(delivered=delivered, charge=c, discharge=d, soc=socs, soc_end=float(soc),
                revenue=float(revenue.sum()), penalty=float(penalty.sum()), degradation=deg_total,
                net=float(revenue.sum() - penalty.sum() - deg_total))


def penalty_over(diff_kwh, cap_kw, price, m: MarketConfig, delivered=None):
    """별표33 방식: 약속보다 허용범위(용량 × tolerance%)를 넘게 '더' 보낸 양에만 페널티.
    모자란 양은 페널티 없이 실시간 가격으로 정산(여기선 같은 가격으로 이미 반영).
    발전량(delivered)이 설비용량의 min_output_ratio 미만인 시간은 면제."""
    allowed = cap_kw * m.tolerance_pct / 100
    over = np.maximum(np.asarray(diff_kwh, dtype=float) - allowed, 0)
    pen = over * np.maximum(price, 0) * m.penalty_ratio
    if delivered is not None:
        pen = np.where(np.asarray(delivered, dtype=float) < cap_kw * m.min_output_ratio, 0.0, pen)
    return pen


# ---------------------------------------------------------------------------
# 1년 성과 리포트: 매일 '전날 정보로 입찰표 → 실제 발전·가격으로 정산'을 1년 반복
#   - 탭 1(입찰표)·탭 5(재현)와 같은 규칙: 시장별 실제 가격, 별표33(넘침만 페널티)
#   - 배터리 상태는 다음 날로 이어짐
#   - 세 가지 비교: 배터리 없음 / 감가 무시 계획(배터리를 공짜라고 보고 계획) / Voltics(감가 고려)
#     ※ 감가 무시 계획도 정산할 때는 실제로 닳은 만큼 비용을 뺀다 → '막 쓰면 얼마나 손해인가'
# ---------------------------------------------------------------------------
def _expected_price_fast(mat, daily_cf, D, target_cf, k=10, window=30):
    past = mat.index[mat.index < D]
    pool = past.intersection(daily_cf.index)
    pool = pool[_doy_window(pool, D, window)] if len(pool) else pool
    if len(pool) >= k:
        diff = (daily_cf.loc[pool] - target_cf).abs().sort_values()
        return mat.loc[diff.index[:k]].mean().values
    recent = past[past >= D - pd.Timedelta(days=7)]
    return mat.loc[recent].mean().values if len(recent) else mat.loc[past].mean().values


def rule_plan(gen_forecast, b: BatteryConfig, cap_kw, soc_start=0.5, reserve=0.1,
              charge_hours=range(10, 15), discharge_hours=range(18, 22)):
    """현장에서 흔한 단순 규칙: 낮(10~14시)엔 발전량으로 최대한 충전, 저녁(18~21시)엔 최대한 방전.
    가격·감가는 보지 않음. 입찰은 '예측 − 충전 계획 + 방전 계획' 수량으로."""
    g = np.asarray(gen_forecast, dtype=float)
    E = usable_energy(b, cap_kw)
    P = b.power_kw if b.power_kw is not None else E * b.c_rate
    eta = np.sqrt(b.efficiency)
    soc = soc_start
    c, d = np.zeros(len(g)), np.zeros(len(g))
    for t in range(len(g)):
        if E <= 0:
            break
        if t in charge_hours:
            c[t] = min(P, g[t], max((b.soc_max - soc) * E / eta, 0))
            soc += c[t] * eta / E
        elif t in discharge_hours:
            d[t] = min(P, max((soc - b.soc_min - reserve) * E * eta, 0))
            soc -= d[t] / eta / E
    return c, d


def backtest_year(hcf, prices_long, cap_kw, b: BatteryConfig, m: MarketConfig, reserve=0.1,
                  start=None, end=None, margin=20.0):
    """hcf: date, hour, cf_pred, cf_act (지역 평균 이용률)
    비교하는 네 가지:
      none     : 배터리 없이 AI 예측대로 입찰
      naive_da : 전날 '예측 가격'으로 배터리 계획, 감가 무시 (막 쓰기)
      aware_da : 전날 '예측 가격'으로 배터리 계획, 감가 + 안전 여유(margin원/kWh) 고려
      aware_rt : 당일 '확정된 하루전 가격'을 보고 배터리 운영, 감가 고려 (⚡ 실시간 탭 방식)
    모두 실제 발전량·실제 가격으로 정산하고, 실제로 닳은 만큼 감가비용을 뺀다."""
    from dataclasses import replace
    mat = prices_long.pivot_table(index="date", columns="hour", values="price").reindex(columns=range(24))
    mat = mat.dropna(thresh=20).ffill(axis=1).bfill(axis=1)
    daily = hcf.groupby("date")[["cf_pred", "cf_act"]].mean()
    days = sorted(set(daily.index) & set(mat.index))
    if start is not None:
        days = [d for d in days if d >= pd.Timestamp(start)]
    if end is not None:
        days = [d for d in days if d <= pd.Timestamp(end)]
    piv_p = hcf.pivot_table(index="date", columns="hour", values="cf_pred").reindex(columns=range(24)).fillna(0)
    piv_a = hcf.pivot_table(index="date", columns="hour", values="cf_act").reindex(columns=range(24)).fillna(0)

    plan_cfg = {
        "naive_da": replace(b, capex_won_per_kwh=1.0),
        "aware_da": replace(b, capex_won_per_kwh=b.capex_won_per_kwh + margin * 2 * b.cycle_life * 0.8),
        "aware_rt": b,
    }
    soc = {k: (b.soc_min + b.soc_max) / 2 for k in list(plan_cfg) + ["curve"]}
    rows = []
    for D in days:
        pred = piv_p.loc[D].values * cap_kw
        act = piv_a.loc[D].values * cap_kw
        price = mat.loc[D].values
        exp_p = _expected_price_fast(mat, daily["cf_act"], D, daily.loc[D, "cf_pred"])
        none = settle_actual(pred, act, np.zeros(24), np.zeros(24), price, cap_kw, b, m)
        row = {"date": D, "none": none["net"], "none_pen": none["penalty"]}
        for k, bb in plan_cfg.items():
            p_plan = price if k == "aware_rt" else exp_p
            plan, _ = plan_day(pred, p_plan, bb, cap_kw, soc[k], reserve)
            r = settle_actual(plan["bid_kwh"].values, act, plan["charge_kwh"].values,
                              plan["discharge_kwh"].values, price, cap_kw, b, m, soc[k])
            soc[k] = r["soc_end"]
            row[k] = r["net"]
            row[f"{k}_pen"] = r["penalty"]
            row[f"{k}_deg"] = r["degradation"]
            row[f"{k}_moved"] = float(r["charge"].sum() + r["discharge"].sum())
        # 단순 규칙 (낮 충전·저녁 방전)
        rc, rd = rule_plan(pred, b, cap_kw, soc.get("rule", (b.soc_min + b.soc_max) / 2), reserve)
        r = settle_actual(pred - rc + rd, act, rc, rd, price, cap_kw, b, m, soc.get("rule", (b.soc_min + b.soc_max) / 2))
        soc["rule"] = r["soc_end"]
        row["rule"], row["rule_pen"], row["rule_deg"] = r["net"], r["penalty"], r["degradation"]
        row["rule_moved"] = float(r["charge"].sum() + r["discharge"].sum())
        # Voltics 가격-수량 입찰 (전날 정보만으로 조건부 주문)
        cv = bid_curve(pred, exp_p, b, cap_kw, soc["curve"], reserve)
        r = settle_curve(cv, act, price, b, cap_kw, m, soc["curve"], reserve)
        soc["curve"] = r["soc_end"]
        row["curve"], row["curve_pen"], row["curve_deg"] = r["net"], r["penalty"], r["degradation"]
        row["curve_moved"] = float(r["charge"].sum() + r["discharge"].sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# 가격-수량 입찰표 (bid curve): 시간마다 "이 가격 이상이면 이만큼 팔겠다"를 최대 3구간으로
#   구간 ① 기본 판매  : 태양광 예측량 중 배터리에 담지 않을 몫. 0원 이상이면 판다 (마이너스면 출력 줄임)
#   구간 ② 담을 몫    : 가격이 '담아 두는 게 나은 가격(기준가)' 이상이면 팔고, 그보다 싸면 팔지 않고 배터리에 담는다
#   구간 ③ 배터리 판매 : 가격이 '꺼내 팔 만한 가격(기준가)' 이상이면 배터리에서 꺼내 판다
#   → 내일 가격을 정확히 몰라도 된다. 가격이 정해지는 순간 시장이 알아서 고르는 '조건부 주문'이라서.
# 기준가 계산 (배터리 1kWh 감가 = c, 왕복효율 = η²):
#   담은 전기의 가치 V = (꺼내 팔 시간의 예상 가격 − c)
#   ② 담기 기준가  = η²·V − c          (지금 가격이 이보다 싸면 파는 것보다 담는 게 이득)
#   ③ 꺼내기 기준가 = c + (그 뒤 시간에 꺼내 팔 가치)  (뒤로 갈수록 낮아짐)
# ---------------------------------------------------------------------------
def bid_curve(gen_forecast, exp_prices, b: BatteryConfig, cap_kw, soc_start=0.5, reserve=0.1, curtail_hours=()):
    """curtail_hours: 출력제어가 예고된 시간 → 그 시간 발전량은 팔 수 없으니 배터리에 담는 걸 최우선으로"""
    g = np.asarray(gen_forecast, dtype=float)
    p = np.asarray(exp_prices, dtype=float)
    n = len(g)
    E = usable_energy(b, cap_kw)
    P = b.power_kw if b.power_kw is not None else E * b.c_rate
    eta2 = b.efficiency
    c = plan_deg_cost(b, cap_kw, min(b.soc_min + reserve, b.soc_max), b.soc_max, P)   # 비선형 감가(구간·속도 평균)
    E_room = max(E * (b.soc_max - b.soc_min - reserve), 0.0)          # 하루에 굴릴 수 있는 에너지

    q_base, q_store, q_dis = g.copy(), np.zeros(n), np.zeros(n)
    p_base = np.zeros(n)
    p_store, p_dis = np.full(n, np.nan), np.full(n, np.nan)
    curt = set(int(h) for h in curtail_hours)
    for h in curt:                                   # 출력제어: 기본 판매 안 함
        q_base[h], p_base[h] = 0.0, np.inf
    thr_store = np.nan
    if E > 0 and P > 0 and E_room > 0:
        # 꺼내 팔 시간: 예상 가격이 높은 순서로, 하루 굴릴 에너지만큼
        dis_hours, cum = [], 0.0
        for h in np.argsort(-p):
            if cum >= E_room * np.sqrt(eta2):
                break
            dis_hours.append(h)
            cum += P
        dis_hours = sorted(dis_hours)
        V = max(np.mean(p[dis_hours]) - c, 0.0)
        thr_store = eta2 * V - c
        # 담을 시간: 출력제어 시간 먼저(버려질 전기라 담는 게 거의 항상 이득), 그다음 예상 가격이 낮은 순서로
        if thr_store > 0:
            cum = 0.0
            order = sorted([h for h in curt if g[h] > 0]) + sorted([h for h in range(n) if g[h] > 0 and h not in curt],
                                                                    key=lambda h: p[h])
            for h in order:
                if cum >= E_room / np.sqrt(eta2):
                    break
                q = min(P, g[h], E_room / np.sqrt(eta2) - cum)
                q_store[h], p_store[h] = q, (np.inf if h in curt else thr_store)
                if h not in curt:
                    q_base[h] -= q
                cum += q
        # 꺼내기 기준가: 감가 + (그 뒤 꺼내 팔 시간의 가치 중 최고)
        dis_hours = [h for h in dis_hours if h not in curt]
        # 하한: 오늘 못 팔아도 내일로 넘겨 팔 수 있으니, 그 가치의 절반 아래로는 팔지 않음
        #       (마지막 시간 기준가가 감가만 남아 '싸게 팔아 손해'가 나던 문제 보완)
        carry = 0.5 * V
        for i, h in enumerate(dis_hours):
            later = [p[k] - c for k in dis_hours[i + 1:]]
            p_dis[h] = c + max(max(later) if later else 0.0, carry, 0.0)
            q_dis[h] = P
    return pd.DataFrame({"q_base": q_base, "p_base": p_base, "q_store": q_store, "p_store": p_store,
                         "q_dis": q_dis, "p_dis": p_dis,
                         "v_store": thr_store if (thr_store == thr_store and thr_store > 0) else np.nan,
                         "curtail": [h in curt for h in range(n)]})


def settle_curve(curve: pd.DataFrame, actual_gen, price, b: BatteryConfig, cap_kw, m: MarketConfig,
                 soc_start=0.5, reserve=0.1):
    """가격이 정해지면 각 구간이 팔렸는지(낙찰) 보고, 실제 발전량·배터리 상태로 정산.
    - 기본 판매가 안 팔린 시간(마이너스 가격·출력제어)은 약속량 넘게 못 보냄 → 남는 전기는 배터리에 담고 나머지는 출력 감축
    - 담는 게 이득인지는 '담은 전기의 가치(v_store)'와 지금 가격을 비교"""
    E = usable_energy(b, cap_kw)
    P = b.power_kw if b.power_kw is not None else E * b.c_rate
    eta = np.sqrt(b.efficiency)
    n = len(price)
    soc = min(max(soc_start, 0.0), 1.0)
    p_base = np.asarray(curve["p_base"], dtype=float) if "p_base" in curve else np.zeros(n)
    v_store = float(curve["v_store"].iloc[0]) if "v_store" in curve and curve["v_store"].notna().any() else np.nan
    bid, c, d, delivered, socs = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(n):
        pr = price[t]
        base_ok = pr >= p_base[t]
        sold_base = curve.q_base[t] if base_ok else 0.0
        sold_store = curve.q_store[t] if (curve.q_store[t] > 0 and pr >= curve.p_store[t]) else 0.0
        want_dis = curve.q_dis[t] if (curve.q_dis[t] > 0 and pr >= curve.p_dis[t]) else 0.0
        if E > 0:
            can_take = curve.q_store[t] - sold_store                       # 안 팔린 '담을 몫'
            if not base_ok and v_store == v_store and pr < v_store:        # 기본 판매도 안 팔렸으면 그것도 담기
                can_take += curve.q_base[t]
            if can_take > 0:
                room = max((b.soc_max - soc) * E / eta, 0.0)
                c[t] = min(can_take, actual_gen[t], room, P)
                soc += c[t] * eta / E
            avail = max((soc - b.soc_min - reserve) * E * eta, 0.0)
            d[t] = min(want_dis, avail, P)
            soc -= d[t] / eta / E
        bid[t] = sold_base + sold_store + d[t]                     # 꺼낼 수 없는 몫은 애초에 약속 안 한 걸로
        out = actual_gen[t] - c[t] + d[t]
        if not base_ok:                                            # 마이너스 가격·출력제어: 약속보다 더 보내지 않게 출력 감축
            out = min(out, bid[t])
        delivered[t] = out
        socs[t] = soc
    revenue = delivered * price
    penalty = penalty_over(delivered - bid, cap_kw, price, m, delivered)
    deg_total = actual_deg_cost(b, cap_kw, c, d, socs, soc_start)
    return dict(bid=bid, delivered=delivered, charge=c, discharge=d, soc=socs, soc_end=float(soc),
                revenue=float(revenue.sum()), penalty=float(penalty.sum()), degradation=deg_total,
                net=float(revenue.sum() - penalty.sum() - deg_total))


def curve_table(curve: pd.DataFrame, gen_forecast, exp_prices, b: BatteryConfig, cap_kw, soc_start=0.5, reserve=0.1):
    """화면·CSV용 입찰표 + '예상 가격대로 낙찰되면' 어떻게 되는지"""
    exp = settle_curve(curve, np.asarray(gen_forecast, float), np.asarray(exp_prices, float), b, cap_kw,
                       MarketConfig(), soc_start, reserve)
    def f(x):
        x = np.asarray(x, dtype=float)
        return np.where(np.isfinite(x), np.round(x, 1), np.nan)
    t = pd.DataFrame({
        "시간": [f"{h:02d}시" for h in range(len(curve))],
        "AI 예측 발전량(kWh)": np.round(gen_forecast, 0),
        "예상 가격(원/kWh)": np.round(exp_prices, 1),
        "①기본판매 가격(원)": f(curve["p_base"].where(curve["q_base"] > 0).values),
        "①기본판매 수량(kWh)": curve["q_base"].round(0),
        "②담을몫 가격(원)": f(curve["p_store"].values),
        "②담을몫 수량(kWh)": curve["q_store"].round(0),
        "③배터리판매 가격(원)": f(curve["p_dis"].values),
        "③배터리판매 수량(kWh)": curve["q_dis"].round(0),
        "예상 낙찰량(kWh)": np.round(exp["bid"], 0),
        "예상 SOC(%)": np.round(exp["soc"] * 100, 0),
        "출력제어": np.where(curve["curtail"] if "curtail" in curve else False, "Y", ""),
        "담기 가치(원/kWh)": f(curve["v_store"].values if "v_store" in curve else np.full(len(curve), np.nan)),
    })
    return t, exp


def fallback_generation(hist: pd.DataFrame, sites, cap_kw, target_date, window_days=20) -> pd.DataFrame:
    """[백업] 기상청 예보를 못 받을 때: 과거 같은 계절(±window_days) 실측 발전의 시간대별 중앙값으로 내일 발전량 추정"""
    h = hist[hist["site"].isin(sites)].copy()
    h["cf"] = h["gen_kwh"] / h["cap_kw"]
    h = h[_doy_window(h["datetime"].dt.normalize(), target_date, window_days)]
    prof = h.groupby(h["datetime"].dt.hour)["cf"].median().reindex(range(24)).fillna(0)
    d0 = pd.Timestamp(target_date)
    return pd.DataFrame({"datetime": d0 + pd.to_timedelta(range(24), unit="h"), "pred_kwh": prof.values * cap_kw})
