"""
Voltics 배터리 엔진 — 배터리 수명 감가를 고려한 임밸런스(예측 오차) 보정 시뮬레이터

아이디어 (쉽게):
  - 하루 전에 '입찰량(=예측 × 보정계수)'을 낸다.
  - 실제 발전량이 입찰량과 다르면 페널티를 낸다.
  - 남으면 배터리에 충전, 모자라면 배터리에서 방전해서 오차를 메운다.
  - 그런데 배터리는 쓸수록 닳는다(감가). 그래서
        '아끼는 페널티' > '배터리가 닳는 비용'  일 때만 배터리를 쓴다.  ← 수명 감가 고려형

세 가지 전략을 같은 데이터로 비교한다.
  1) 배터리 없음
  2) 단순 운전   : 오차가 나면 무조건 배터리 사용
  3) 감가 고려   : 감가비용보다 이득일 때만 사용 + SOC(충전량)가 너무 높거나 낮은 구간은 피함

※ 페널티 규칙은 경영 파트 정산식이 나오기 전까지 쓰는 '가정'이다. penalty_cost() 한 곳만 바꾸면 된다.
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
    stress_factor: float = 2.0       # 위 구간에서 감가비용 배수


@dataclass
class MarketConfig:
    bid_factor: float = 1.0          # 입찰량 = 예측 × bid_factor  (모델이 흐린 날 과대예측 → 1보다 작게 하면 유리할 수 있음)
    tolerance_pct: float = 8.0       # 설비용량 대비 이 % 이내 오차는 페널티 없음
    penalty_ratio: float = 1.0       # 허용범위를 넘는 오차 1kWh당 페널티 = SMP × penalty_ratio


# ---------------------------------------------------------------------------
# 비용 계산식
# ---------------------------------------------------------------------------
def base_degradation_cost(b: BatteryConfig) -> float:
    """배터리로 1kWh를 움직일 때(충전 또는 방전) 닳는 비용(원/kWh).
    설치비 ÷ (평생 충방전 가능한 총량). 1사이클 = 충전 1번 + 방전 1번."""
    dod = 0.8
    return b.capex_won_per_kwh / (2 * b.cycle_life * dod)


def stress_multiplier(soc: float, b: BatteryConfig) -> float:
    """충전량이 너무 높거나 낮은 상태에서 쓰면 더 빨리 닳는다."""
    return b.stress_factor if (soc < b.stress_low or soc > b.stress_high) else 1.0


def penalty_cost(deviation_kwh, cap_kw, smp, m: MarketConfig):
    """[가정] 허용범위(용량 × tolerance%)를 넘는 오차에 SMP × penalty_ratio 만큼 페널티.
    경영 파트의 KPX 정산식이 나오면 이 함수만 바꾸면 된다."""
    allowed = cap_kw * m.tolerance_pct / 100
    excess = np.maximum(np.abs(deviation_kwh) - allowed, 0)
    return excess * smp * m.penalty_ratio


# ---------------------------------------------------------------------------
# 시뮬레이션
# ---------------------------------------------------------------------------
STRATEGIES = {"none": "배터리 없음", "naive": "단순 운전", "aware": "감가 고려 (Voltics)"}


def simulate(df: pd.DataFrame, b: BatteryConfig, m: MarketConfig, strategy: str) -> pd.DataFrame:
    """df 컬럼: datetime, cap_kw, gen_kwh(실제), pred_kwh(예측), smp(원/kWh)
    시간 순서대로 한 줄씩 배터리를 운전한다."""
    df = df.sort_values("datetime").reset_index(drop=True)
    cap = float(df["cap_kw"].iloc[0])
    E = usable_energy(b, cap)             # 배터리 용량 (kWh)
    P = b.power_kw if b.power_kw is not None else E * b.c_rate   # 시간당 최대 충방전량 (kWh)
    eta = np.sqrt(b.efficiency)           # 충전·방전 각각의 효율
    c0 = base_degradation_cost(b)
    allowed = cap * m.tolerance_pct / 100

    bid = df["pred_kwh"].to_numpy() * m.bid_factor
    act = df["gen_kwh"].to_numpy()
    smp = df["smp"].to_numpy()
    n = len(df)

    soc = b.soc_init
    out_soc, out_ch, out_dis, out_deg = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    use_battery = strategy != "none" and E > 0

    for t in range(n):
        dev = act[t] - bid[t]                       # + 남음 / - 모자람
        excess = abs(dev) - allowed                 # 페널티가 붙는 양
        ch = dis = deg = 0.0

        if use_battery and excess > 0:
            avoided = smp[t] * m.penalty_ratio      # 1kWh 메울 때 아끼는 페널티
            if dev > 0:                             # 남는다 → 충전
                room = (b.soc_max - soc) * E / eta
                amount = min(excess, P, max(room, 0))
                if strategy == "aware":
                    # 고스트레스 구간(너무 꽉 참)에 들어가지 않는 만큼만, 그리고 이득일 때만
                    safe = max((b.stress_high - soc) * E / eta, 0)
                    cost_safe = c0
                    cost_stress = c0 * b.stress_factor
                    amt_safe = min(amount, safe) if avoided > cost_safe else 0.0
                    amt_stress = (amount - min(amount, safe)) if avoided > cost_stress else 0.0
                    amount = amt_safe + amt_stress
                    deg = amt_safe * cost_safe + amt_stress * cost_stress
                else:
                    deg = amount * c0 * stress_multiplier(soc, b)
                ch = amount
                soc += ch * eta / E
            else:                                   # 모자란다 → 방전
                avail = (soc - b.soc_min) * E * eta
                amount = min(excess, P, max(avail, 0))
                if strategy == "aware":
                    safe = max((soc - b.stress_low) * E * eta, 0)
                    cost_safe = c0
                    cost_stress = c0 * b.stress_factor
                    amt_safe = min(amount, safe) if avoided > cost_safe else 0.0
                    amt_stress = (amount - min(amount, safe)) if avoided > cost_stress else 0.0
                    amount = amt_safe + amt_stress
                    deg = amt_safe * cost_safe + amt_stress * cost_stress
                else:
                    deg = amount * c0 * stress_multiplier(soc, b)
                dis = amount
                soc -= dis / eta / E

        # 여유 충전: 페널티가 안 붙는 범위(허용오차) 안에서 발전량 일부를 미리 배터리에 채워 둔다.
        #   단순 운전 → 최고 충전량까지 꽉 채움 / 감가 고려 → 열화가 적은 중간 구간(stress_high)까지만
        if use_battery and act[t] > 0:
            target = b.stress_high if strategy == "aware" else b.soc_max
            margin = (act[t] - ch + dis - bid[t]) + allowed      # 이만큼까지는 덜 팔아도 페널티 없음
            room = (target - soc) * E / eta
            rec = max(min(P - ch - dis, room, margin, act[t] - ch), 0.0)
            if strategy == "aware" and smp[t] * m.penalty_ratio <= c0:
                rec = 0.0                                          # 나중에 아낄 페널티가 감가비용보다 작으면 안 채움
            if rec > 0:
                deg += rec * c0 * (1.0 if strategy == "aware" else stress_multiplier(soc, b))
                ch += rec
                soc += rec * eta / E

        out_soc[t], out_ch[t], out_dis[t], out_deg[t] = soc, ch, dis, deg

    delivered = act - out_ch + out_dis
    remaining_dev = delivered - bid
    res = df.copy()
    res["bid_kwh"] = bid
    res["charge_kwh"] = out_ch
    res["discharge_kwh"] = out_dis
    res["soc"] = out_soc
    res["delivered_kwh"] = delivered
    res["revenue"] = delivered * smp
    res["penalty"] = penalty_cost(remaining_dev, cap, smp, m)
    res["degradation"] = out_deg
    res["net"] = res["revenue"] - res["penalty"] - res["degradation"]
    return res


def usable_energy(b: BatteryConfig, cap_kw: float) -> float:
    rated = b.energy_kwh if b.energy_kwh is not None else cap_kw * b.hours
    return rated * b.soh


def summarize(res: pd.DataFrame, b: BatteryConfig) -> dict:
    cap = float(res["cap_kw"].iloc[0])
    E = usable_energy(b, cap)
    throughput = res["charge_kwh"].sum() + res["discharge_kwh"].sum()
    cycles = throughput / (2 * E) if E > 0 else 0.0
    # 수명 소모: cycle_life(80% DoD) 사이클을 다 쓰면 용량 100% → 80%
    soh_drop = cycles / (b.cycle_life * 0.8) * 20 if E > 0 else 0.0
    return {
        "판매수익(원)": res["revenue"].sum(),
        "페널티(원)": res["penalty"].sum(),
        "배터리 감가비용(원)": res["degradation"].sum(),
        "순수익(원)": res["net"].sum(),
        "등가 사이클(회)": cycles,
        "수명 소모(%p)": soh_drop,
    }


def compare(df: pd.DataFrame, b: BatteryConfig, m: MarketConfig):
    """세 전략을 모두 돌려서 (요약표, 전략별 시간별 결과) 반환"""
    runs = {k: simulate(df, b, m, k) for k in STRATEGIES}
    table = pd.DataFrame({STRATEGIES[k]: summarize(r, b) for k, r in runs.items()}).T
    return table, runs


def scale_to_plant(df: pd.DataFrame, my_cap_kw: float) -> pd.DataFrame:
    """지역 데이터(수십 MW)를 사업자 발전소 용량에 맞게 비율로 줄이거나 늘린다.
    같은 지역이면 날씨가 비슷하니 '용량 대비 발전 비율(이용률)'은 같다고 본다."""
    k = my_cap_kw / float(df["cap_kw"].iloc[0])
    out = df.copy()
    out["cap_kw"] = my_cap_kw
    out["gen_kwh"] *= k
    out["pred_kwh"] *= k
    return out


# 배터리 예시값 (실제 제품 사양이 아니라 일반적인 범위를 담은 시작값 — 사업자가 자기 값으로 바꿔 쓴다)
PRESETS = {
    "직접 입력": None,
    "LFP 표준형 (예시)":   dict(duration_h=2.0, efficiency=0.88, cycle_life=6000, capex=300_000, soh=1.00),
    "NMC 고출력형 (예시)": dict(duration_h=1.0, efficiency=0.90, cycle_life=4000, capex=350_000, soh=1.00),
    "사용 중 배터리 (예시)": dict(duration_h=2.0, efficiency=0.85, cycle_life=6000, capex=300_000, soh=0.85),
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
                b: BatteryConfig, soc_now: float, m: MarketConfig):
    E = usable_energy(b, cap_kw)
    P = b.power_kw if b.power_kw is not None else E * b.c_rate
    eta = np.sqrt(b.efficiency)
    c_deg = base_degradation_cost(b)
    allowed = cap_kw * m.tolerance_pct / 100
    future_best = max(future_prices) if len(future_prices) else 0.0

    dev = expected_kwh - promised_kwh                     # + 남음 / - 모자람
    can_dis = max(min(P, (soc_now - b.soc_min) * E * eta), 0.0)   # 지금 꺼낼 수 있는 양(계통에 나가는 양)
    can_ch = max(min(P, (b.soc_max - soc_now) * E / eta), 0.0)    # 지금 넣을 수 있는 양(발전소에서 빼는 양)

    def settle(delivered):
        """약속 대비 정산(이번 시간): 약속분 수익은 이미 확정이므로 '차이'에 대한 돈만 계산"""
        d = delivered - promised_kwh
        money = d * price_now                                   # 모자라면 -(사 오는 돈), 남으면 +(추가 판매)
        over = max(d - allowed, 0.0)                            # 별표33: 약속보다 허용범위 넘게 더 보내면 페널티
        return money - over * max(price_now, 0) * m.penalty_ratio

    options = []

    # ① 대기
    options.append(dict(action="대기", kwh=0.0, now=settle(expected_kwh), later=0.0, deg=0.0,
                        why="배터리를 쓰지 않고 차이는 실시간 가격으로 정산해요."))

    # ② 방전: 배터리 전기를 지금 쓴다. 대신 오늘 남은 시간 중 가장 비쌀 때 쓸 기회를 잃는다
    #    후보: 모자란 만큼만 메우기 / 허용범위까지 더 보내기
    if can_dis > 0:
        best_d = None
        for x in sorted({min(can_dis, max(-dev, 0.0)), min(can_dis, max(-dev, 0.0) + allowed)}):
            if x <= 0:
                continue
            o = dict(action="방전", kwh=x, now=settle(expected_kwh + x), later=-x * future_best, deg=-x * c_deg,
                     why=f"지금 {price_now:,.0f}원일 때 쓰는 대신, 오늘 남은 시간 최고가 {future_best:,.0f}원에 쓸 기회를 포기해요.")
            o["total"] = o["now"] + o["later"] + o["deg"]
            if best_d is None or o["total"] > best_d["total"]:
                best_d = o
        if best_d:
            options.append(best_d)

    # ③ 충전: 발전한 전기 일부를 지금 팔지 않고 담아 뒀다가 비쌀 때 판다
    #    후보: 약속보다 남는 만큼만 / 넣을 수 있는 만큼 최대
    if can_ch > 0 and expected_kwh > 0:
        best_c = None
        for y in sorted({min(can_ch, max(dev, 0.0)), min(can_ch, expected_kwh)}):
            if y <= 0:
                continue
            out = y * eta * eta                                  # 나중에 실제로 꺼낼 수 있는 양 (효율 손실)
            o = dict(action="충전", kwh=y, now=settle(expected_kwh - y), later=out * future_best,
                     deg=-(y + out) * c_deg,
                     why=f"지금 {price_now:,.0f}원에 팔지 않고 담아 뒀다가 {future_best:,.0f}원일 때 팔아요 "
                         f"(효율 손실 {1 - b.efficiency:.0%}).")
            o["total"] = o["now"] + o["later"] + o["deg"]
            if best_c is None or o["total"] > best_c["total"]:
                best_c = o
        if best_c:
            options.append(best_c)

    for o in options:
        o["total"] = o["now"] + o["later"] + o["deg"]
    table = pd.DataFrame(options)
    best = table.loc[table["total"].idxmax()]
    info = dict(dev=dev, c_deg=c_deg, future_best=future_best, allowed=allowed,
                can_dis=can_dis, can_ch=can_ch)
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
    return long.sort_values(["date", "hour"]).reset_index(drop=True)
