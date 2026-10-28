import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import battery as bt
from predictor import (DATA_DIR, asos_market, is_jeju, latlon_to_grid, load_model, predict_location,
                       reference_sites, tomorrow_forecast)

st.set_page_config(page_title="Voltics | AI 전력 거래 엔진", page_icon="⚡", layout="wide")


# ---------- 한 번만 불러오고 재사용 ----------
@st.cache_resource
def get_model():
    return load_model()


@st.cache_data(ttl=1800, show_spinner="기상청 예보를 받아오는 중...")
def get_forecast(key, nx, ny):
    return tomorrow_forecast(key, nx, ny)


@st.cache_data
def get_metrics():
    return pd.read_csv(DATA_DIR / "metrics_by_site.csv", encoding="utf-8-sig")


PRICE_FILES = {"제주": DATA_DIR / "jeju_smp_hourly.csv", "육지": DATA_DIR / "land_smp_hourly.csv"}
MARKET_HELP = (
    "전기 가격은 시장마다 달라요. 제주 발전소는 제주 가격, 육지 발전소는 육지 가격으로 팔아요. "
    "지도에서 고른 위치에 맞는 가격을 자동으로 써요. "
    "참고: 2026년 현재 재생에너지 입찰제도는 제주에서만 시범 시행 중이에요(전력시장운영규칙 별표 33). "
    "육지 결과는 제도가 육지로 확대될 때를 대비한 시나리오로 봐 주세요."
)
EPSIS_URL = "https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShdChart.do?menuId=040202"
ACTION_ICON = {"충전": "🔋⬆️", "방전": "🔋⬇️", "대기": "⏸️"}
TOL_HELP = ("약속(급전지시)보다 이 범위를 넘게 더 보내면 임밸런스 페널티가 붙어요. "
            "전력시장운영규칙 [별표 33] 제주 시범사업 정산기준 3.가.(2)(라) 기준 8% (2025년, 2024년까지는 12%). "
            "시기에 따라, 또는 입찰제도가 육지로 확대되면서 기준이 달라질 수 있으니 최신 규칙에 맞게 설정하세요.")


@st.cache_data(ttl=6 * 3600, show_spinner="기상청 ASOS 과거 일사량을 받아오는 중...")
def get_asos(key, market):
    return asos_market(key, market)


@st.cache_data
def load_hist():
    f = DATA_DIR / "site_hourly_2025.csv"
    return pd.read_csv(f, encoding="utf-8-sig", parse_dates=["datetime"]) if f.exists() else None


@st.cache_data
def get_daily_cf(sites):
    f = DATA_DIR / "site_hourly_2025.csv"
    if not f.exists():
        return None
    hist = pd.read_csv(f, encoding="utf-8-sig", parse_dates=["datetime"])
    return bt.daily_capacity_factor(hist, list(sites))


@st.cache_data
def load_price_file(raw_bytes):
    import io
    return bt.read_epsis_hourly(io.BytesIO(raw_bytes))


def get_key():
    try:
        return st.secrets["KMA_KEY"]
    except Exception:
        return ""


model, meta = get_model()

if "loc" not in st.session_state:
    st.session_state.loc = (33.4996, 126.5312)   # 처음 위치: 제주시 (입찰제도 시행 지역)

st.title("⚡ Voltics")
st.markdown("**배터리 수명 감가 고려형 AI 전력 거래 엔진** · 재생에너지 입찰제도 대응")
st.caption("① AI가 내일 발전량을 예측하고 → ② 예측 오차는 배터리로 메우되, 배터리가 닳는 비용까지 따져서 가장 이득인 운전을 고릅니다.")

# ---------- 왼쪽 입력창 ----------
with st.sidebar:
    st.header("내 발전소")
    capacity = st.number_input("설비용량 (kW)", min_value=1.0, value=1000.0, step=100.0)
    lat, lon = st.session_state.loc
    nx, ny = latlon_to_grid(lat, lon)
    st.caption(f"📍 위치: 위도 {lat:.4f}, 경도 {lon:.4f}")
    st.caption(f"기상청 예보 칸 번호: ({nx}, {ny}) — 전국을 5km 네모칸으로 나눈 번호예요")
    st.divider()
    st.header("내 배터리")
    preset_name = st.selectbox("배터리 유형", list(bt.PRESETS),
                           help="예시값은 일반적인 범위의 시작값이에요. 보유한 배터리의 사양서 값으로 바꿔 넣으세요.")
    ps = bt.PRESETS[preset_name] or dict(duration_h=1.0, efficiency=0.90, cycle_life=6000, capex=300_000, soh=1.0)
    wkey = f"_{preset_name}"   # 유형을 바꾸면 입력칸 기본값도 바뀌도록
    energy = st.number_input("배터리 용량 (kWh)", 0.0, 400_000.0, float(round(capacity * ps["duration_h"])), 100.0,
                             key="e" + wkey)
    power = st.number_input("최대 충·방전 출력 (kW)", 0.0, 200_000.0,
                            float(round(energy / ps["duration_h"])) if energy else 0.0, 100.0, key="p" + wkey)
    efficiency = st.slider("왕복 효율", 0.70, 0.98, ps["efficiency"], 0.01, key="ef" + wkey)
    soh = st.slider("현재 건강도 SOH (%)", 50, 100, int(ps["soh"] * 100), 1, key="soh" + wkey,
                    help="새 배터리는 100%. 오래 써서 용량이 줄었으면 낮춰 주세요.") / 100
    capex = st.number_input("배터리 구입·설치비 (원/kWh)", 50_000, 2_000_000, ps["capex"], 10_000, key="c" + wkey)
    cycle_life = st.number_input("수명 사이클 (회, 80% DoD 기준)", 500, 20_000, ps["cycle_life"], 500, key="cl" + wkey)
    soc_range = st.slider("사용할 충전량 범위 (%)", 0, 100, (10, 90), 5, key="soc" + wkey)

    st.divider()
    st.header("전기 가격", help=MARKET_HELP)
    market = "제주" if is_jeju(lat, lon) else "육지"
    if market == "제주":
        st.success("📍 제주 발전소 → **제주 가격** 적용 · 입찰제도 시행 지역")
    else:
        st.warning("📍 육지 발전소 → **육지 가격** 적용 · 입찰제도 **확대 대비 시나리오**", icon="🧭")
    st.link_button("📈 전력거래소 시간별 SMP 바로가기", EPSIS_URL, use_container_width=True)
    st.caption("구분(제주/육지)과 기간(**2023년 1월 ~ 오늘** 권장)을 고르고 **조회 → CSV**로 받아서 맞는 칸에 올려 주세요.")
    all_prices = {}
    for mk in ["제주", "육지"]:
        up = st.file_uploader(f"{mk} 시간별 SMP CSV", type=["csv"], key=f"up_{mk}")
        if up is not None:
            all_prices[mk] = load_price_file(up.getvalue())
        elif PRICE_FILES[mk].exists():
            all_prices[mk] = load_price_file(PRICE_FILES[mk].read_bytes())
    prices = all_prices.get(market)
    for mk, pr in all_prices.items():
        st.caption(f"{mk} 가격: {pr['date'].min():%Y-%m-%d} ~ {pr['date'].max():%Y-%m-%d}" + (" ← 사용 중" if mk == market else ""))
    if prices is None or not len(prices):
        st.caption(f"⚠️ {market} 가격 데이터가 없어서 모든 시간을 100원으로 가정해요.")

bcfg = bt.BatteryConfig(energy_kwh=energy, power_kw=power, soh=soh, efficiency=efficiency,
                        capex_won_per_kwh=capex, cycle_life=cycle_life,
                        soc_min=soc_range[0] / 100, soc_max=soc_range[1] / 100)


tab1, tab4, tab5, tab3, tab2 = st.tabs(
    ["☀️ 내일 발전량 예측", "⚡ 실시간 운영 판단", "⏪ 과거 날짜 재현", "📈 1년 성과 리포트", "📊 모델 성능"])

# ---------- 탭 1: 내일 예측 ----------
with tab1:
    st.markdown("##### 📍 지도에서 발전소 위치를 클릭하세요")
    fmap = folium.Map(location=[36.0, 127.8], zoom_start=7)
    folium.Marker(st.session_state.loc, tooltip="내 발전소").add_to(fmap)
    clicked = st_folium(fmap, height=380, use_container_width=True, returned_objects=["last_clicked"])
    if clicked and clicked.get("last_clicked"):
        new_loc = (round(clicked["last_clicked"]["lat"], 4), round(clicked["last_clicked"]["lng"], 4))
        if new_loc != st.session_state.loc:
            st.session_state.loc = new_loc
            st.rerun()

    forecast, day = None, None
    key = get_key()
    if not key:
        st.warning("기상청 API 키가 없어요. Streamlit 앱 설정의 Secrets에 KMA_KEY를 넣어 주세요.")
    else:
        try:
            forecast, day = get_forecast(key, nx, ny)
        except Exception as e:
            st.error(f"예보를 받아오지 못했어요: {e}")

    if forecast is not None and len(forecast) == 0:
        st.error("내일 예보가 비어 있어요. 잠시 후 다시 시도해 주세요.")
    elif forecast is not None:
        res, ref_sites = predict_location(model, meta, capacity, forecast, lat, lon)
        total = res["pred_kwh"].sum()
        peak = res.loc[res["pred_kwh"].idxmax()]

        st.subheader(f"{capacity:,.0f} kW 발전소 · {day} 예측")
        if market == "육지":
            st.info(
                "🧭 **육지 확대 대비 시나리오**: 2026년 현재 재생에너지 입찰제도는 제주에서만 시범 시행 중이에요. "
                "육지 발전소 결과는 제도가 육지로 확대될 때 Voltics가 어떻게 쓰일지 보여 주는 참고용이에요. "
                "(육지는 2026년 3월부터 재생에너지 준중앙급전이 시작되는 등 제도 확대가 진행 중이에요.)"
            )
        c1, c2, c3 = st.columns(3)
        c1.metric("하루 총 예상 발전량", f"{total:,.0f} kWh")
        c2.metric("가장 많이 나오는 시간", f"{peak['datetime'].hour}시 · {peak['pred_kwh']:,.0f} kWh")
        c3.metric("하루 이용률", f"{total / (capacity * 24) * 100:.1f} %")

        chart = res.assign(시각=res["datetime"].dt.hour).set_index("시각")[["pred_kwh"]]
        st.bar_chart(chart.rename(columns={"pred_kwh": "예측 발전량 (kWh)"}))

        m = get_metrics()
        rows = m[(m["지역"].isin(ref_sites)) & (m["모델"] == "B_예보대응")]
        if not rows.empty:
            st.caption(
                f"참고: 이 위치는 학습 데이터 중 {', '.join(ref_sites)} 발전소 특성을 기준으로 예측했어요. "
                f"이 지역들에서 모델의 평균 오차는 설비용량 대비 약 {rows['용량기준 오차율(%)'].mean():.1f}%이고, "
                "흐린 날에는 실제보다 많게 예측하는 경향이 있어요."
            )

        with st.expander("시간별 자세히 보기"):
            table = forecast.merge(res, on="datetime")
            table.columns = ["시각", "기온(℃)", "습도(%)", "풍속(m/s)", "강수(mm)", "예측 발전량(kWh)"]
            st.dataframe(table.round(1), use_container_width=True, hide_index=True)
            st.download_button(
                "CSV로 받기", table.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"예측_{nx}_{ny}_{day}.csv", mime="text/csv",
            )

        # ---------- 연결 ①: AI 예측 + 배터리 → 내일 추천 입찰표 ----------
        st.divider()
        st.subheader("🧾 내일 가격-수량 입찰표 (AI 예측 + 배터리)")
        st.caption(
            "시간마다 **\"이 가격 이상이면 이만큼 팔겠다\"**를 구간별로 내는 입찰표예요. "
            "예를 들어 \"한낮에 가격이 ○원보다 싸면 팔지 않고 배터리에 담고, 저녁에 ○원 이상이면 배터리에서 꺼내 판다\"처럼 "
            "**조건을 걸어 두기 때문에 내일 가격을 정확히 맞히지 않아도** 가격이 정해지는 순간 유리한 쪽으로 낙찰돼요. "
            "기준가에는 배터리가 닳는 비용과 효율 손실이 들어가 있어요."
        )
        tomorrow_cf = float(res["pred_kwh"].sum() / (capacity * 24))
        if prices is not None and len(prices):
            dcf = get_daily_cf(tuple(ref_sites))
            pinfo = dict(method="recent")
            # 1순위: ASOS 실측 일사량으로 2023~어제 전체에서 비슷한 날 찾기
            asos_err = None
            try:
                sun = get_asos(key, market)
                target_pct = bt.season_percentile(tomorrow_cf, dcf, day) if dcf is not None else 0.5
                exp_price, pinfo = bt.similar_day_prices_asos(prices, sun["gsr"], target_pct, day)
            except Exception as e:
                asos_err = str(e)
            # 2순위: 2025년 실측 발전량으로 비슷한 날 찾기 / 3순위: 최근 7일 평균
            if pinfo["method"] != "asos":
                if dcf is not None:
                    exp_price, pinfo = bt.similar_day_prices(prices, dcf, tomorrow_cf, day)
                else:
                    exp_price, pinfo = bt.expected_prices(prices, days=7), dict(method="recent")

            if pinfo["method"] == "asos":
                price_note = (
                    f"예상 가격 = **기상청 ASOS 실측 일사량** 기준으로 내일과 맑은 정도가 비슷했던 과거 {pinfo['n']}일의 가격 모양 "
                    f"× 최근 28일 평균 가격 수준({pinfo['recent_level']:,.0f}원). "
                    f"내일은 이 계절 기준 **맑은 정도 상위 {100 - pinfo['target_pct'] * 100:.0f}%** 예상이에요."
                )
            elif pinfo["method"] == "similar":
                price_note = (
                    f"예상 가격 = **내일과 햇빛 양이 비슷했던 2025년 {pinfo['n']}일**의 시간대별 평균 가격 "
                    f"(내일 AI 예측 이용률 {pinfo['target_cf']:.1%} ↔ 고른 날들 실측 평균 {pinfo['cf_mean']:.1%}, 같은 계절 기준)"
                )
            else:
                price_note = f"예상 가격 = 최근 7일 시간대별 평균 ({prices['date'].max():%m/%d}까지)"
            if asos_err:
                price_note += (" · ℹ️ ASOS 일사량을 못 받아서 2025년 데이터로 대신 계산했어요. "
                               "공공데이터포털에서 '기상청_지상(종관, ASOS) 일자료 조회서비스'를 활용신청해 주세요.")
            if pinfo.get("days"):
                with st.expander("가격 예측에 쓴 과거 날짜 보기"):
                    st.write(", ".join(f"{d:%Y-%m-%d}" for d in pinfo["days"]))
        else:
            exp_price = pd.Series([100.0] * 24, index=range(24))
            price_note = "⚠️ 가격 데이터가 없어 모든 시간 100원으로 가정 → 시간대별 가격 차이가 없어서 배터리를 거의 안 써요"

        # 연결 ③: AI가 흐린 날 많게 예측하는 버릇 → 비 예보가 있으면 배터리를 더 남겨 둠
        daytime = forecast[(forecast["datetime"].dt.hour >= 7) & (forecast["datetime"].dt.hour <= 18)]
        rainy = bool((daytime["rain"] > 0).any() or (daytime["hum"] >= 85).mean() > 0.5)
        r1, r2 = st.columns(2)
        soc_start = r1.slider("내일 0시 배터리 충전량 (%)", 0, 100, 50, 5) / 100
        reserve = r2.slider(
            "예비 충전량 (%)", 0, 50, 20 if rainy else 5, 5,
            help="입찰에 쓰지 않고 남겨 두는 양이에요. AI는 흐린 날 실제보다 많게 예측하는 경향이 있어서, "
                 "모자랄 때 당일 메우려고 남겨 둬요. 비·높은 습도 예보가 있으면 자동으로 20%로 올려요.",
        ) / 100
        if rainy:
            st.info("🌧️ 내일 낮에 비나 높은 습도가 예보돼 있어요. AI가 과대예측할 수 있어서 예비 충전량을 늘렸어요.")

        pred_arr = res["pred_kwh"].values
        curve = bt.bid_curve(pred_arr, exp_price.values, bcfg, capacity, soc_start, reserve)
        tbl, exp_r = bt.curve_table(curve, pred_arr, exp_price.values, bcfg, capacity, soc_start, reserve)
        plan = pd.DataFrame({"gen_kwh": pred_arr, "bid_kwh": exp_r["bid"], "soc": exp_r["soc"]}, index=range(24))
        plan["soc_before"] = np.r_[soc_start, exp_r["soc"][:-1]]
        st.session_state["bid_plan"] = plan
        st.session_state["bid_curve"] = curve
        st.session_state["bid_reserve"] = reserve
        st.session_state["bid_soc_start"] = soc_start
        st.session_state["tomorrow_price"] = pd.Series(exp_price.values, index=range(24))

        # 조건 요약 (사람이 읽는 문장)
        c_deg = bt.base_degradation_cost(bcfg)
        st_h = [h for h in range(24) if curve["q_store"][h] > 0]
        ds_h = [h for h in range(24) if curve["q_dis"][h] > 0]
        if st_h or ds_h:
            lines = []
            if st_h:
                lines.append(f"🔋⬆️ **{', '.join(f'{h}시' for h in st_h)}**: 가격이 **{curve['p_store'][st_h[0]]:,.0f}원보다 싸면** "
                             f"그 시간 발전량 중 최대 {curve['q_store'][st_h].max():,.0f} kWh를 팔지 않고 배터리에 담아요.")
            if ds_h:
                thr = ", ".join(f"{h}시 {curve['p_dis'][h]:,.0f}원" for h in ds_h)
                lines.append(f"🔋⬇️ 배터리에서 꺼내 팔 조건 (이 가격 **이상**이면 최대 {curve['q_dis'][ds_h].max():,.0f} kWh): {thr}")
            lines.append("☀️ 나머지 발전량은 **0원 이상이면 판매**, 마이너스 가격이면 출력을 줄여요.")
            st.markdown("\n\n".join(lines))
        else:
            st.info(
                "이번 입찰표엔 배터리 조건을 걸지 않았어요. 예상 가격 차이로는 담았다 꺼내 팔아도 "
                f"효율 손실({1 - efficiency:.0%})과 배터리 감가(1kWh당 {c_deg:,.1f}원)를 넘기 어렵다고 봤어요."
            )

        base_exp = float((pred_arr * exp_price.values).sum())
        volt_exp = exp_r["revenue"] - exp_r["degradation"]
        k1, k2, k3 = st.columns(3)
        k1.metric("배터리 없이 (예상 가격 기준)", f"{base_exp / 10_000:,.1f}만 원")
        k2.metric("가격-수량 입찰 (예상 가격 기준)", f"{volt_exp / 10_000:,.1f}만 원", f"{(volt_exp - base_exp) / 10_000:+,.1f}만 원",
                  help="예상 가격대로 낙찰된다고 쳤을 때의 수입 − 배터리 감가. 실제 가격이 조건에 맞으면 더 벌 수 있어요.")
        k3.metric("배터리 감가비용 (예상)", f"{exp_r['degradation'] / 10_000:,.1f}만 원")
        st.caption(price_note)

        hrs = [f"{h:02d}시" for h in range(24)]
        cdf = pd.DataFrame({"시간": hrs, "AI 예측 발전량": pred_arr, "예상 낙찰량": exp_r["bid"]})
        bars = alt.Chart(cdf).mark_bar(color="#9cc3ea", opacity=0.8).encode(
            x=alt.X("시간:N", sort=hrs, title=None), y=alt.Y("AI 예측 발전량:Q", title="kWh"),
            tooltip=["시간", alt.Tooltip("AI 예측 발전량:Q", format=",.0f"), alt.Tooltip("예상 낙찰량:Q", format=",.0f")])
        line = alt.Chart(cdf).mark_line(color="#e8590c", strokeWidth=3, point=True).encode(
            x=alt.X("시간:N", sort=hrs), y="예상 낙찰량:Q")
        st.altair_chart(bars + line, use_container_width=True)
        st.caption("🟦 막대 = AI 예측 발전량 · 🟧 선 = 예상 가격대로 정해지면 팔게 될 양. 선이 막대보다 낮으면 배터리에 담고, 높으면 꺼내 팔아요.")

        with st.expander("시간별 가격-수량 입찰표 보기 · 받기"):
            st.dataframe(tbl, use_container_width=True, hide_index=True)
            st.download_button("입찰표 CSV로 받기", tbl.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"가격수량입찰표_{day}.csv", mime="text/csv")
            st.caption("실제 전력거래소 입찰은 시간마다 최대 10구간까지 가격-수량을 낼 수 있어요. Voltics는 3구간(기본 판매·담을 몫·배터리 판매)을 써요.")
        st.caption("내일이 되면 '⚡ 실시간 운영 판단' 탭에 이 CSV를 올리세요. 확정된 가격으로 각 시간 낙찰량(약속량)을 자동 계산해 줘요.")

# ---------- 탭 2: 모델 성능 ----------
with tab2:
    m = get_metrics()
    st.subheader("지역별 오차율 (2025년 시험기간)")
    st.caption("용량기준 오차율 = |예측 − 실제| ÷ 설비용량. 낮을수록 좋아요. 예측제도 정산 기준은 8% 이하예요.")
    err = m.pivot(index="지역", columns="모델", values="용량기준 오차율(%)")
    err.columns = ["모델 A (관측 기반)", "모델 B (예보 대응·웹 사용)"]
    st.bar_chart(err, stack=False)
    st.dataframe(m, use_container_width=True, hide_index=True)


# ---------- 탭 3: 1년 성과 리포트 (탭 1·4·5와 같은 규칙) ----------
def region_hourly_cf(hist_all, sites):
    """과거 실측 → 날짜·시간별 이용률 (AI 예측 / 실제), 여러 지역이면 평균"""
    hh = hist_all[hist_all["site"].isin(sites)].copy()
    hh["date"] = hh["datetime"].dt.normalize()
    hh["hour"] = hh["datetime"].dt.hour
    hh["cf_pred"] = hh["pred_kwh"] / hh["cap_kw"]
    hh["cf_act"] = hh["gen_kwh"] / hh["cap_kw"]
    return hh.groupby(["date", "hour"])[["cf_pred", "cf_act"]].mean().reset_index()


@st.cache_data(show_spinner="1년치 입찰·정산을 하루씩 다시 계산하는 중... (몇 초 걸려요)")
def run_backtest(hcf, prices_long, cap_kw, b_kwargs, tol, reserve, margin, start, end):
    return bt.backtest_year(hcf, prices_long, cap_kw, bt.BatteryConfig(**b_kwargs),
                            bt.MarketConfig(tolerance_pct=tol), reserve, start, end, margin)


def won(x):
    return f"{x / 10_000:+,.0f}만 원" if abs(x) >= 5_000 else "±0만 원"


with tab3:
    st.subheader("이 엔진을 2025년 1년 동안 썼다면? (성과 리포트)")
    st.caption(
        "매일 **전날 알 수 있었던 정보로 입찰** → **그날 실제 발전량·실제 가격으로 정산**을 1년 동안 반복했어요. "
        "배터리 상태는 다음 날로 이어지고, 실제로 닳은 만큼 감가비용을 빼요. "
        f"위치·용량·배터리·가격({market})은 왼쪽 사이드바 설정을 그대로 써요."
    )
    hist_all3 = load_hist()
    if hist_all3 is None or prices is None or not len(prices):
        st.warning("성과 리포트를 만들려면 과거 발전 데이터(site_hourly_2025.csv)와 사이드바의 가격 데이터가 필요해요.")
    else:
        rs3 = reference_sites(lat, lon, meta)
        hcf3 = region_hourly_cf(hist_all3, rs3)
        dmin, dmax = hcf3["date"].min().date(), hcf3["date"].max().date()
        c1, c2, c3, c4 = st.columns(4)
        period = c1.date_input("기간", (dmin, dmax), min_value=dmin, max_value=dmax)
        reserve3 = c2.slider("예비 충전량 (%)", 0, 50, 10, 5, key="bt_res") / 100
        tol3 = c3.slider("페널티 없는 초과 범위 (%)", 0.0, 20.0, 8.0, 0.5, key="bt_tol", help=TOL_HELP)
        margin3 = 20
        c4.caption("배터리 사양·위치·가격은 왼쪽 사이드바에서 바꿔요.")
        if isinstance(period, tuple) and len(period) == 2:
            b_kwargs = dict(energy_kwh=energy, power_kw=power, soh=soh, efficiency=efficiency,
                            capex_won_per_kwh=capex, cycle_life=cycle_life,
                            soc_min=soc_range[0] / 100, soc_max=soc_range[1] / 100)
            r3 = run_backtest(hcf3, prices, capacity, b_kwargs, tol3, reserve3, margin3,
                              pd.Timestamp(period[0]), pd.Timestamp(period[1]))
            tot = r3.sum()
            n_days = len(r3)
            st.markdown(f"#### {capacity:,.0f} kW · 배터리 {energy:,.0f} kWh / {power:,.0f} kW · "
                        f"{', '.join(rs3)} 날씨 · {market} 가격 · {n_days}일")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("배터리 없이 (AI 예측대로)", f"{tot['none'] / 1e8:,.2f}억 원", help="실제 판매수입 − 페널티")
            k2.metric("수량만 입찰 · 감가 무시", won(tot["naive_da"] - tot["none"]),
                      help="전날 예측 가격으로 배터리 계획을 세우고 수량만 입찰. 배터리를 공짜처럼 막 씀")
            k3.metric("⚡ Voltics 가격-수량 입찰", won(tot["curve"] - tot["none"]),
                      help="전날 정보만으로 '이 가격이면 담고, 이 가격이면 꺼내 판다'는 조건부 입찰. 감가 고려")
            k4.metric("참고: 가격을 미리 알았다면", won(tot["aware_rt"] - tot["none"]),
                      help="당일 확정 가격을 미리 안다고 쳤을 때의 상한선 (현실에선 불가능한 기준값)")

            best_txt = won(tot["curve"] - tot["naive_da"]).replace("+", "")
            ratio = (tot["curve"] - tot["none"]) / (tot["aware_rt"] - tot["none"]) if tot["aware_rt"] > tot["none"] else float("nan")
            st.success(
                f"배터리를 **감가 생각 없이 막 쓰면 {won(tot['naive_da'] - tot['none'])}**인데, "
                f"Voltics **가격-수량 입찰은 {won(tot['curve'] - tot['none'])}**이에요. 같은 배터리로 **{best_txt}** 차이. "
                + (f"전날 정보만으로 '가격을 미리 알았을 때' 성과의 **{ratio:.0%}**를 냈어요." if ratio == ratio else "")
            )

            st.markdown("##### 누적 수익 차이 (배터리 없음 대비, 만 원)")
            cum = pd.DataFrame({
                "수량만 입찰 · 감가 무시": (r3["naive_da"] - r3["none"]).cumsum(),
                "⚡ Voltics 가격-수량 입찰": (r3["curve"] - r3["none"]).cumsum(),
                "참고: 가격을 미리 알았다면": (r3["aware_rt"] - r3["none"]).cumsum(),
            }) / 10_000
            st.line_chart(cum)

            keys = ["naive_da", "aware_da", "curve", "aware_rt"]
            detail = pd.DataFrame({
                "순수익 차이(만 원)": [0] + [(tot[k] - tot.none) / 1e4 for k in keys],
                "초과 페널티(만 원)": [tot.none_pen / 1e4] + [tot[f"{k}_pen"] / 1e4 for k in keys],
                "배터리 감가(만 원)": [0] + [tot[f"{k}_deg"] / 1e4 for k in keys],
                "배터리 쓴 날(일)": [0] + [int((r3[f"{k}_moved"] > 1).sum()) for k in keys],
            }, index=["배터리 없이", "수량만 입찰 · 감가 무시", "수량만 입찰 · 감가+안전여유",
                      "⚡ Voltics 가격-수량 입찰", "참고: 가격을 미리 알았다면"])
            st.dataframe(detail.style.format("{:,.0f}"), use_container_width=True)

            with st.expander("이 숫자를 어떻게 봐야 하나요? (가정과 한계)"):
                st.markdown(
                    "- **수량만 입찰하면 손해인 이유**: 전날엔 내일 가격을 '예측'만 할 수 있어서, 예측이 빗나간 날 배터리를 써서 손해를 봐요.\n"
                    "- **가격-수량 입찰이 이기는 이유**: '이 가격보다 싸면 담고, 이 가격 이상이면 꺼내 판다'는 조건을 걸어 두면, "
                    "가격을 맞히지 않아도 실제 가격이 정해지는 순간 유리한 쪽으로 낙찰돼요.\n"
                    "- **정산 가정**: 하루전·실시간 가격을 같은 시간별 SMP로 보고, 약속보다 허용범위를 넘게 더 보낸 양에만 페널티(별표 33 방식). "
                    "실제 실시간 가격은 더 출렁여서, 배터리로 메우는 가치는 이보다 클 수 있어요.\n"
                    "- **입찰 가정**: 하루전 시장에서 시간별 가격-수량 구간 입찰이 가능하고, 배터리(보조자원) 방전분도 입찰량에 넣을 수 있다고 봤어요. "
                    "이 부분은 규칙 확인이 필요해요.\n"
                    "- **AI 예측값**: 팀원 모델 B의 2025년 시험 결과(그날 관측된 기온·습도·바람·비 입력)라 실제 전날 예보보다 조금 정확할 수 있어요."
                )

# ---------- 탭 4: 실시간 운영 판단 ----------

def render_realtime():
    st.subheader("지금 이 시간, 배터리를 어떻게 할까?")
    st.caption(
        "하루 전 약속한 양(낙찰량)과 지금 예상되는 발전량이 다를 때, "
        "**충전 · 방전 · 대기** 중 돈이 가장 많이 남는 쪽을 배터리가 닳는 비용까지 빼고 골라 줘요."
    )

    now_kst = pd.Timestamp.now(tz="Asia/Seoul")
    today = now_kst.date()
    plan = st.session_state.get("bid_plan")
    tp = st.session_state.get("tomorrow_price")
    mode = st.radio(
        "운영 날짜", ["오늘 실제 운영", "내일 입찰표로 미리 연습"], horizontal=True,
        help="오늘 실제 운영: 오늘 확정된 하루전 가격과 어제 받아 둔 입찰표로 판단해요.\n\n"
             "내일 연습: 탭 1에서 만든 내일 입찰표와 내일 예상 가격으로 미리 연습해요.",
    )

    # --- ① 가격표 ---
    st.markdown("#### ① 가격표")
    st.caption(f"가격 데이터는 왼쪽 사이드바 '전기 가격'에서 올려요. 지금은 **{market} 가격**을 쓰고 있어요.")
    if mode == "오늘 실제 운영":
        if prices is not None and len(prices):
            dates = sorted(prices["date"].dt.date.unique())
            if today in dates:
                day_pick = today
            else:
                day_pick = dates[-1]
                st.warning(f"가격 파일에 오늘({today}) 가격이 아직 없어서 가장 최근인 **{day_pick}** 가격을 보여 줘요. "
                           "EPSIS에서 오늘 가격까지 받아 올려 주세요.")
            day_prices = (prices[prices["date"].dt.date == day_pick].set_index("hour")["price"]
                          .reindex(range(24)).ffill().bfill())
            price_label = f"{day_pick} 하루전 가격 (EPSIS)"
        else:
            st.info("가격 파일이 없어서 100원으로 시작해요. 아래 표에 직접 넣어도 돼요.")
            day_prices = pd.Series([100.0] * 24, index=range(24))
            price_label = "직접 입력"
    else:
        if tp is None:
            st.info("☀️ 탭 1에서 내일 예측을 먼저 하면, 내일 입찰표와 예상 가격으로 연습할 수 있어요.")
            return
        day_prices = tp.copy()
        price_label = "내일 예상 가격 (탭 1)"
    hour = st.selectbox("판단할 시간", list(range(24)), index=int(now_kst.hour) if mode == "오늘 실제 운영" else 12,
                        format_func=lambda h: f"{h:02d}시 ~ {h + 1:02d}시")

    with st.expander(f"가격표 보기·고치기 (원/kWh) · {price_label}"):
        edited = st.data_editor(
            pd.DataFrame({"시간": [f"{h:02d}시" for h in range(24)], "가격": day_prices.values}),
            hide_index=True, use_container_width=True, key=f"pt_{mode}_{hour}",
        )
        day_prices = pd.Series(edited["가격"].astype(float).values, index=range(24))

    price_now = st.number_input(
        f"⚡ 지금({hour:02d}시) 실시간 가격 (원/kWh)", -100.0, 1000.0, float(day_prices[hour]), 1.0,
        key=f"pn_{mode}_{hour}_{day_prices[hour]}",
        help="기본값은 가격표(EPSIS 시간별 SMP, 전날 미리 정해진 하루전 기준 가격)의 이 시간 값이에요. "
             "당일 실제 실시간 가격이 이와 다르면 새 값을 넣어 주세요. 판단이 실시간 가격 기준으로 바뀌어요.",
    )
    st.caption(
        "ℹ️ 위 가격은 **전날 미리 정해진 하루전 기준 가격**이에요. 당일 실시간 가격이 달라졌다면 "
        "**새 가격을 직접 입력**해 주세요. (제주 실시간시장 가격은 15분마다 다시 정해져요.)"
    )

    # 앞으로 비교할 가격: 이 날 남은 시간 (+ 오늘 운영이면 내일 0~8시 예상 가격)
    TOMORROW_UNTIL = 9   # 내일 9시 전까지만: 해가 뜨면 태양광으로 다시 채울 수 있어서
    d_lbl = "오늘" if mode == "오늘 실제 운영" else "내일"
    fut_today = day_prices[day_prices.index > hour]
    future = pd.Series(fut_today.values, index=[f"{d_lbl} {h:02d}시" for h in fut_today.index], dtype=float)
    use_next = mode == "오늘 실제 운영" and tp is not None
    if use_next:
        fut_tmr = tp[tp.index < TOMORROW_UNTIL]
        future = pd.concat([future, pd.Series(fut_tmr.values, index=[f"내일 {h:02d}시" for h in fut_tmr.index])])

    # --- ② 이번 시간 상황 ---
    st.markdown("#### ② 이번 시간 상황")
    src = None
    cv_use, gen_use, soc0_use = None, None, 0.5
    if mode == "오늘 실제 운영":
        up_plan = st.file_uploader("어제 받아 둔 '가격-수량 입찰표' CSV (있으면 약속량을 자동으로 계산해요)", type=["csv"], key="up_plan")
        if up_plan is not None:
            try:
                import io
                tb = pd.read_csv(io.BytesIO(up_plan.getvalue()), encoding="utf-8-sig")
                cv_use = pd.DataFrame({
                    "q_base": tb["①기본판매 수량(kWh)"].fillna(0).values, "p_base": 0.0,
                    "q_store": tb["②담을몫 수량(kWh)"].fillna(0).values, "p_store": tb["②담을몫 가격(원)"].values,
                    "q_dis": tb["③배터리판매 수량(kWh)"].fillna(0).values, "p_dis": tb["③배터리판매 가격(원)"].values,
                })
                gen_use = tb["AI 예측 발전량(kWh)"].fillna(0).values
            except Exception:
                st.error("입찰표 형식을 읽지 못했어요. 탭 1의 '입찰표 CSV로 받기'로 받은 파일인지 확인해 주세요.")
        else:
            st.caption("입찰표 파일이 없으면 약속량을 직접 넣어 주세요. (탭 1에서 '입찰표 CSV로 받기'로 받아 두면 다음 날 여기에 올릴 수 있어요.)")
    elif st.session_state.get("bid_curve") is not None:
        cv_use, gen_use = st.session_state["bid_curve"], plan["gen_kwh"].values
        soc0_use = st.session_state.get("bid_soc_start", 0.5)
    if cv_use is not None:
        # 가격표(확정 가격)로 각 구간이 팔렸는지 계산 → 이 시간 약속량(낙찰량)
        cl = bt.settle_curve(cv_use, gen_use, day_prices.values, bcfg, capacity, bt.MarketConfig(),
                             soc0_use, st.session_state.get("bid_reserve", 0.1))
        src = dict(bid=cl["bid"], gen=gen_use, soc=np.r_[soc0_use, cl["soc"][:-1]] * 100)
        st.caption(f"📄 입찰표를 가격표({price_label})에 맞춰 보니, 이 시간 낙찰량(약속량)은 **{cl['bid'][hour]:,.0f} kWh**예요. "
                   "'지금 예상되는 발전량'을 실제 상황에 맞게 고쳐 보세요.")
    if src is not None:
        d_prom, d_exp, d_soc = float(round(src["bid"][hour])), float(round(src["gen"][hour])), int(round(src["soc"][hour]))
    else:
        d_prom, d_exp, d_soc = 0.0, 0.0, 50
    s1, s2, s3 = st.columns(3)
    promised = s1.number_input("하루 전 약속한 양 (kWh)", 0.0, 1_000_000.0, d_prom, 10.0, key=f"prom_{mode}_{hour}_{d_prom}",
                               help="이 시간에 팔겠다고 낙찰받은 양")
    expected = s2.number_input("지금 예상되는 실제 발전량 (kWh)", 0.0, 1_000_000.0, d_exp, 10.0, key=f"exp_{mode}_{hour}_{d_exp}",
                               help="입찰표의 AI 예측값이 기본값이에요. 현장 계측값이나 최신 예보로 본 값을 넣으세요.")
    soc_now = s3.slider("지금 배터리 충전량 SOC (%)", 0, 100, d_soc, 1, key=f"soc_{mode}_{hour}_{d_soc}") / 100
    t1, t2 = st.columns(2)
    tol = t1.slider("페널티 없는 초과 범위 (설비용량 대비 %)", 0.0, 20.0, 8.0, 0.5, help=TOL_HELP)
    res_default = int(round(st.session_state.get("bid_reserve", 0.1) * 100))
    reserve_now = t2.slider(
        "예비 충전량 (%)", 0, 50, res_default, 5, key=f"rt_res_{res_default}",
        help="약속하지 않은 전기를 배터리에서 꺼내 더 팔 때도 이만큼은 남겨 둬요(예측이 틀릴 때 메우는 비상용). "
             "모자란 걸 메울 때는 이 예비분까지 써요. 기본값은 탭 1 입찰표의 예비 충전량이에요.",
    ) / 100

    best, opts, info = bt.decide_hour(promised, expected, price_now, future.values, capacity,
                                      bcfg, soc_now, bt.MarketConfig(tolerance_pct=tol), reserve_now)

    # --- ③ 추천 ---
    st.markdown("#### ③ 추천")
    gap = info["dev"]
    gap_txt = f"약속보다 **{abs(gap):,.0f} kWh 모자라요**" if gap < 0 else (
        f"약속보다 **{gap:,.0f} kWh 남아요**" if gap > 0 else "약속과 **딱 맞아요**")
    st.markdown(
        f"{gap_txt} · 지금 가격 **{price_now:,.0f}원** · 앞으로 최고가 **{info['future_best']:,.0f}원** · "
        f"배터리 1kWh 쓸 때 닳는 비용 **{info['c_deg']:,.1f}원**"
    )
    st.caption(
        (f"📈 앞으로 최고가 = **{future.idxmax()} {future.max():,.0f}원** "
         f"({d_lbl} 남은 시간{' + 내일 0~8시 예상 가격' if use_next else ''} 중 가장 비싼 시간). "
         if len(future) else "📈 앞으로 비교할 시간이 없어요. ")
        + ("탭 1에서 내일 예측을 하면 내일 새벽·아침 예상 가격도 같이 비교해요. " if (mode == "오늘 실제 운영" and tp is None) else "")
        + f"🔋 배터리에 남겨 둔 전기 1kWh의 가치 = 앞으로 최고가 {info['future_best']:,.0f}원 − 감가 {info['c_deg']:,.1f}원 "
        f"= **{info['hold_value']:,.1f}원**. 지금 쓰는 게 이보다 이득일 때만 꺼내요. "
        f"약속 안 한 추가 판매는 페널티 없는 범위({info['allowed']:,.0f} kWh) 안에서, 예비 충전량 {info['reserve']:.0%}를 남기고만 해요."
    )
    base = opts.loc[opts["action"] == "대기", "total"].iloc[0]
    msg = f"## {ACTION_ICON[best.action]} {best.action}" + (f" {best.kwh:,.0f} kWh" if best.kwh else "")
    st.success(msg + f"\n\n대기보다 **{best.total - base:,.0f}원** 이득 · {best.why}")
    if gap < 0 and best.action != "방전":
        st.caption("💡 모자라지만 지금 배터리로 메우지 않는 게 나아요. 모자란 만큼은 실시간 가격으로 정산되고, "
                   "배터리 전기는 더 비싼 시간에 쓰는 편이 이득이에요.")

    show = opts[["action", "kwh", "now", "later", "deg", "total"]].copy()
    show.columns = ["선택지", "양(kWh)", "이번 시간 정산(원)", "배터리 속 전기 가치 변화(원)", "지금 닳는 비용(원)", "합계(원)"]
    st.dataframe(show.style.format({c: "{:,.0f}" for c in show.columns[1:]}),
                 use_container_width=True, hide_index=True)
    st.caption("이번 시간 정산 = (실제로 보낸 양 − 약속량) × 지금 가격 − 초과 페널티. "
               "약속량만큼의 수익은 하루 전에 이미 확정돼서 여기선 '차이'만 계산해요.")

    chart = pd.DataFrame({"가격(원/kWh)": day_prices.values}, index=[f"{h:02d}시" for h in range(24)])
    chart.loc[f"{hour:02d}시", "가격(원/kWh)"] = price_now
    st.caption(f"가격 흐름 · {price_label} (지금 시간은 실시간 가격으로 바꿔서 표시)")
    st.bar_chart(chart)

    st.info(
        "제주 시범사업 규칙상 실시간시장 입찰은 1시간 단위이고 **거래시간 75분 전에 마감**돼요. "
        "그래서 실제 운영에서는 '75분 뒤 시간'의 배터리 계획을 미리 정해 변경입찰에 반영하는 식으로 써요."
    )


with tab4:
    render_realtime()


# ---------- 탭 5: 과거 날짜 재현 ----------
def render_replay():
    st.subheader("그날 Voltics를 썼다면?")
    st.caption(
        "2025년의 하루를 골라, **그 전날 알 수 있었던 정보만으로** Voltics가 입찰표를 짜고, "
        "**실제 발전량과 실제 가격**으로 정산해 봐요. 배터리 없이 AI 예측대로 입찰했을 때와 비교해요."
    )
    hist_all = load_hist()
    if hist_all is None or prices is None or not len(prices):
        st.warning("재현하려면 과거 발전 데이터(site_hourly_2025.csv)와 사이드바의 가격 데이터가 필요해요.")
    else:
        rs = reference_sites(lat, lon, meta)
        hh = hist_all[hist_all["site"].isin(rs)].copy()
        hh["date"] = hh["datetime"].dt.normalize()
        hh["hour"] = hh["datetime"].dt.hour
        hh["cf_pred"] = hh["pred_kwh"] / hh["cap_kw"]
        hh["cf_act"] = hh["gen_kwh"] / hh["cap_kw"]
        hcf = hh.groupby(["date", "hour"])[["cf_pred", "cf_act"]].mean().reset_index()
        pmat = prices.pivot_table(index="date", columns="hour", values="price").reindex(columns=range(24))
        avail = sorted(set(hcf["date"]) & set(pmat.dropna(thresh=20).index))
        if not avail:
            st.warning(f"{market} 가격 데이터에 2025년 날짜가 없어서 재현할 수 없어요. 사이드바에서 2025년이 포함된 가격 파일을 올려 주세요.")
            return

        # 시연 추천 날짜: 한낮 0원이 있었던 맑은 날 / AI가 크게 과대예측한 흐린 날
        dsum = hcf.groupby("date")[["cf_pred", "cf_act"]].mean()
        zero_days = [d for d in avail if (pmat.loc[d, 10:14] <= 0).any()]
        sunny_zero = dsum.loc[zero_days].sort_values("cf_act", ascending=False).index[:3] if zero_days else []
        over = (dsum.loc[avail, "cf_pred"] - dsum.loc[avail, "cf_act"]).sort_values(ascending=False).index[:2]
        picks = {f"☀️ 맑고 한낮 0원인 날 ({d:%Y-%m-%d})": d for d in sunny_zero}
        picks.update({f"🌧️ AI가 많게 예측한 흐린 날 ({d:%Y-%m-%d})": d for d in over})
        picks["📅 직접 고르기"] = None

        c1, c2 = st.columns([2, 1])
        choice = c1.selectbox("날짜 고르기", list(picks), help=f"{market} 지역 기준으로 시연하기 좋은 날을 골라 뒀어요.")
        if picks[choice] is None:
            d_sel = c1.date_input("날짜", avail[len(avail) // 2].date(), min_value=avail[0].date(), max_value=avail[-1].date())
            D = pd.Timestamp(d_sel)
            if D not in avail:
                st.error("그 날짜는 데이터가 없어요. 다른 날짜를 골라 주세요.")
                return
        else:
            D = picks[choice]
        reserve_r = c2.slider("예비 충전량 (%)", 0, 50, 10, 5, key="rp_res") / 100
        tol_r = c2.slider("페널티 없는 초과 범위 (%)", 0.0, 20.0, 8.0, 0.5, key="rp_tol", help=TOL_HELP)

        day_cf = hcf[hcf["date"] == D].set_index("hour").reindex(range(24)).fillna(0)
        pred = day_cf["cf_pred"].values * capacity
        actual = day_cf["cf_act"].values * capacity
        act_price = pmat.loc[D].values

        # 전날까지의 정보로 가격 예측 (그날 이후 데이터는 안 봄)
        past_prices = prices[prices["date"] < D]
        dcf_all = dsum["cf_act"]
        dcf_past = dcf_all[dcf_all.index < D]
        pred_cf_day = float(pred.sum() / (capacity * 24))
        pinfo = dict(method="recent")
        try:
            if not key:
                raise RuntimeError("no key")
            sun = get_asos(key, market)["gsr"]
            sun = sun[sun.index < D]
            tp = bt.season_percentile(pred_cf_day, dcf_past, D)
            exp_p, pinfo = bt.similar_day_prices_asos(past_prices, sun, tp, D)
        except Exception:
            pass
        if pinfo["method"] != "asos":
            exp_p, pinfo = bt.similar_day_prices(past_prices, dcf_past, pred_cf_day, D, min_overlap=10)
        exp_p = exp_p.reindex(range(24)).ffill().bfill()

        mcfg = bt.MarketConfig(tolerance_pct=tol_r)
        curve_r = bt.bid_curve(pred, exp_p.values, bcfg, capacity, 0.5, reserve_r)
        base_r = bt.settle_actual(pred, actual, np.zeros(24), np.zeros(24), act_price, capacity, bcfg, mcfg)
        volt_r = bt.settle_curve(curve_r, actual, act_price, bcfg, capacity, mcfg, 0.5, reserve_r)

        st.markdown(f"#### {D:%Y-%m-%d} ({'월화수목금토일'[D.dayofweek]}) · {capacity:,.0f} kW · {market}")
        m1, m2, m3 = st.columns(3)
        m1.metric("배터리 없이 AI 예측대로", f"{base_r['net'] / 10_000:,.1f}만 원",
                  help="실제 판매수입 − 페널티")
        m2.metric("Voltics 가격-수량 입찰", f"{volt_r['net'] / 10_000:,.1f}만 원",
                  f"{(volt_r['net'] - base_r['net']) / 10_000:+,.2f}만 원", help="실제 판매수입 − 페널티 − 배터리 감가")
        m3.metric("예측 정확도", f"발전 {abs(pred.sum() - actual.sum()) / max(actual.sum(), 1):.0%} 오차",
                  f"가격 평균 {np.nanmean(np.abs(exp_p.values - act_price)):,.0f}원 오차", delta_color="off")

        hrs = [f"{h:02d}시" for h in range(24)]
        g1, g2 = st.columns(2)
        with g1:
            st.caption("발전량 (kWh): AI 예측 · 실제 발전 · Voltics 낙찰량(실제 가격으로 정해진 약속량)")
            st.line_chart(pd.DataFrame({"AI 예측": pred, "실제 발전": actual,
                                        "Voltics 낙찰량": volt_r["bid"]}, index=hrs))
        with g2:
            st.caption("가격 (원/kWh): 전날 예측 · 실제")
            st.line_chart(pd.DataFrame({"전날 예측 가격": exp_p.values, "실제 가격": act_price}, index=hrs))

        ch_h = [h for h in range(24) if volt_r["charge"][h] > 1]
        dis_h = [h for h in range(24) if volt_r["discharge"][h] > 1]
        st_set = [h for h in range(24) if curve_r["q_store"][h] > 0]
        if ch_h or dis_h:
            st.info(f"🔋 전날 입찰표의 조건대로, 실제 가격이 정해지자 "
                    + (f"**{', '.join(f'{h}시' for h in ch_h)}**에 배터리를 채우고 " if ch_h else "")
                    + (f"**{', '.join(f'{h}시' for h in dis_h)}**에 꺼내 팔았어요." if dis_h else "꺼내 팔 조건은 오지 않았어요."))
        elif st_set:
            st.info(f"🔋 {', '.join(f'{h}시' for h in st_set)}에 '가격이 {curve_r['p_store'][st_set[0]]:,.0f}원보다 싸면 담기' 조건을 걸었지만, "
                    "실제 가격이 그보다 높아서 그냥 팔았어요. (조건부 입찰이라 손해 없이 넘어감)")
        else:
            st.info("🔋 이날은 예상 가격 차이가 배터리 닳는 비용보다 작아서 배터리 조건을 걸지 않았어요.")

        with st.expander("정산 자세히 보기"):
            st.dataframe(pd.DataFrame({
                "배터리 없이": [base_r["revenue"], base_r["penalty"], 0, base_r["net"]],
                "Voltics": [volt_r["revenue"], volt_r["penalty"], volt_r["degradation"], volt_r["net"]],
            }, index=["판매수입(원)", "초과 페널티(원)", "배터리 감가(원)", "순수익(원)"]).style.format("{:,.0f}"),
                use_container_width=True)
            st.caption("정산 가정: 하루전·실시간 가격을 같은 시간별 SMP로 보고, 약속보다 허용범위를 넘게 더 보낸 양에만 페널티(별표 33 방식). "
                       f"가격 예측 방식: {'ASOS 일사량 유사일' if pinfo['method'] == 'asos' else '2025년 발전량 유사일'}. "
                       "AI 예측값은 팀원 모델 B의 2025년 시험 결과(그날 관측된 기온·습도·바람·비 입력)라서, "
                       "실제 전날 예보로 예측했을 때보다 조금 더 정확하게 나올 수 있어요.")


with tab5:
    render_replay()
