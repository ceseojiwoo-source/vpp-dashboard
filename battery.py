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
                b: BatteryConfig, soc_now: float, m: MarketConfig, reserve: float = 0.0):
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
    c_deg = base_degradation_cost(b)
    allowed = cap_kw * m.tolerance_pct / 100
    future_best = max(future_prices) if len(future_prices) else 0.0
    hold_value = max(future_best - c_deg, 0.0)              # 배터리에 남겨 둔 전기 1kWh의 가치

    dev = expected_kwh - promised_kwh                     # + 남음 / - 모자람
    shortfall = max(-dev, 0.0)
    can_dis = max(min(P, (soc_now - b.soc_min) * E * eta), 0.0)              # 최저 충전량까지
    can_dis_extra = max(min(P, (soc_now - b.soc_min - reserve) * E * eta), 0.0)  # 예비 충전량은 남기고
    can_ch = max(min(P, (b.soc_max - soc_now) * E / eta), 0.0)

    def settle(delivered):
        """약속 대비 정산(이번 시간): 약속분 수익은 이미 확정이므로 '차이'에 대한 돈만 계산"""
        d = delivered - promised_kwh
        money = d * price_now                                   # 모자라면 -(사 오는 돈), 남으면 +(추가 판매)
        over = max(d - allowed, 0.0)                            # 별표33: 약속보다 허용범위 넘게 더 보내면 페널티
        return money - over * max(price_now, 0) * m.penalty_ratio

    options = [dict(action="대기", kwh=0.0, now=settle(expected_kwh), later=0.0, deg=0.0,
                    why="배터리를 쓰지 않고 차이는 실시간 가격으로 정산해요.")]

    # 방전 후보: ① 모자란 만큼 메우기  ② 메우고 + 약속 안 한 추가 판매(허용범위 안, 예비 충전량 제외)
    cover = min(can_dis, shortfall)
    extra = max(min(allowed, can_dis_extra - cover, P - cover), 0.0)
    best_d = None
    for x, kind in [(cover, "메우기"), (cover + extra, "추가 판매")]:
        if x <= 0:
            continue
        o = dict(action="방전", kwh=x, now=settle(expected_kwh + x), later=-x * hold_value, deg=-x * c_deg)
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
            o = dict(action="충전", kwh=y, now=settle(expected_kwh - y), later=out * hold_value, deg=-y * c_deg,
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
    info = dict(dev=dev, c_deg=c_deg, future_best=future_best, hold_value=hold_value, allowed=allowed,
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
    c_deg = base_degradation_cost(b)
    soc0 = min(max(soc_start, 0.0), 1.0)
    # 하한 = 최저 충전량 + 예비분. 단, 지금 충전량이 그보다 낮으면 '지금보다 더 쓰지 않기'까지만 요구
    lo = min(b.soc_min + reserve, b.soc_max, soc0)
    hi = max(b.soc_max, soc0)

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
    penalty = penalty_over(delivered - bid, cap_kw, price, m)
    deg = (c + d) * base_degradation_cost(b)
    return dict(delivered=delivered, charge=c, discharge=d, soc=socs, soc_end=float(soc),
                revenue=float(revenue.sum()), penalty=float(penalty.sum()), degradation=float(deg.sum()),
                net=float(revenue.sum() - penalty.sum() - deg.sum()))


def penalty_over(diff_kwh, cap_kw, price, m: MarketConfig):
    """별표33 방식: 약속보다 허용범위(용량 × tolerance%)를 넘게 '더' 보낸 양에만 페널티.
    모자란 양은 페널티 없이 실시간 가격으로 정산(여기선 같은 가격으로 이미 반영)."""
    allowed = cap_kw * m.tolerance_pct / 100
    over = np.maximum(np.asarray(diff_kwh) - allowed, 0)
    return over * np.maximum(price, 0) * m.penalty_ratio


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
def bid_curve(gen_forecast, exp_prices, b: BatteryConfig, cap_kw, soc_start=0.5, reserve=0.1):
    g = np.asarray(gen_forecast, dtype=float)
    p = np.asarray(exp_prices, dtype=float)
    n = len(g)
    E = usable_energy(b, cap_kw)
    P = b.power_kw if b.power_kw is not None else E * b.c_rate
    eta2 = b.efficiency
    c = base_degradation_cost(b)
    E_room = max(E * (b.soc_max - b.soc_min - reserve), 0.0)          # 하루에 굴릴 수 있는 에너지

    q_base, q_store, q_dis = g.copy(), np.zeros(n), np.zeros(n)
    p_store, p_dis = np.full(n, np.nan), np.full(n, np.nan)
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
        # 담을 시간: 발전하는 시간 중 예상 가격이 낮은 순서로
        if thr_store > 0:
            cum = 0.0
            for h in sorted([h for h in range(n) if g[h] > 0], key=lambda h: p[h]):
                if cum >= E_room / np.sqrt(eta2):
                    break
                q = min(P, g[h], E_room / np.sqrt(eta2) - cum)
                q_store[h], p_store[h] = q, thr_store
                q_base[h] -= q
                cum += q
        # 꺼내기 기준가: 감가 + (그 뒤 꺼내 팔 시간의 가치 중 최고)
        for i, h in enumerate(dis_hours):
            later = [p[k] - c for k in dis_hours[i + 1:]]
            p_dis[h] = c + max(max(later) if later else 0.0, 0.0)
            q_dis[h] = P
    return pd.DataFrame({"q_base": q_base, "p_base": 0.0, "q_store": q_store, "p_store": p_store,
                         "q_dis": q_dis, "p_dis": p_dis})


def settle_curve(curve: pd.DataFrame, actual_gen, price, b: BatteryConfig, cap_kw, m: MarketConfig,
                 soc_start=0.5, reserve=0.1):
    """가격이 정해지면 각 구간이 팔렸는지(낙찰) 보고, 실제 발전량·배터리 상태로 정산"""
    E = usable_energy(b, cap_kw)
    eta = np.sqrt(b.efficiency)
    n = len(price)
    soc = min(max(soc_start, 0.0), 1.0)
    bid, c, d, delivered, socs = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(n):
        pr = price[t]
        sold_base = curve.q_base[t] if pr >= curve.p_base[t] else 0.0
        sold_store = curve.q_store[t] if (curve.q_store[t] > 0 and pr >= curve.p_store[t]) else 0.0
        want_dis = curve.q_dis[t] if (curve.q_dis[t] > 0 and pr >= curve.p_dis[t]) else 0.0
        if E > 0:
            if curve.q_store[t] > 0 and sold_store == 0:          # 담을 몫이 안 팔림 → 배터리에 담기
                room = max((b.soc_max - soc) * E / eta, 0.0)
                c[t] = min(curve.q_store[t], actual_gen[t], room)
                soc += c[t] * eta / E
            avail = max((soc - b.soc_min - reserve) * E * eta, 0.0)
            d[t] = min(want_dis, avail)
            soc -= d[t] / eta / E
        bid[t] = sold_base + sold_store + d[t]                     # 꺼낼 수 없는 몫은 애초에 약속 안 한 걸로
        out = actual_gen[t] - c[t] + d[t]
        if pr < 0:                                                 # 마이너스 가격: 약속보다 더 보내지 않게 출력 줄임
            out = min(out, bid[t])
        delivered[t] = out
        socs[t] = soc
    revenue = delivered * price
    penalty = penalty_over(delivered - bid, cap_kw, price, m)
    deg = (c + d) * base_degradation_cost(b)
    return dict(bid=bid, delivered=delivered, charge=c, discharge=d, soc=socs, soc_end=float(soc),
                revenue=float(revenue.sum()), penalty=float(penalty.sum()), degradation=float(deg.sum()),
                net=float(revenue.sum() - penalty.sum() - deg.sum()))


def curve_table(curve: pd.DataFrame, gen_forecast, exp_prices, b: BatteryConfig, cap_kw, soc_start=0.5, reserve=0.1):
    """화면·CSV용 입찰표 + '예상 가격대로 낙찰되면' 어떻게 되는지"""
    exp = settle_curve(curve, np.asarray(gen_forecast, float), np.asarray(exp_prices, float), b, cap_kw,
                       MarketConfig(), soc_start, reserve)
    def f(x):
        return np.where(np.isnan(x), np.nan, np.round(x, 1))
    t = pd.DataFrame({
        "시간": [f"{h:02d}시" for h in range(len(curve))],
        "AI 예측 발전량(kWh)": np.round(gen_forecast, 0),
        "예상 가격(원/kWh)": np.round(exp_prices, 1),
        "①기본판매 가격(원)": curve["p_base"].where(curve["q_base"] > 0).round(1),
        "①기본판매 수량(kWh)": curve["q_base"].round(0),
        "②담을몫 가격(원)": f(curve["p_store"].values),
        "②담을몫 수량(kWh)": curve["q_store"].round(0),
        "③배터리판매 가격(원)": f(curve["p_dis"].values),
        "③배터리판매 수량(kWh)": curve["q_dis"].round(0),
        "예상 낙찰량(kWh)": np.round(exp["bid"], 0),
        "예상 SOC(%)": np.round(exp["soc"] * 100, 0),
    })
    return t, exp
