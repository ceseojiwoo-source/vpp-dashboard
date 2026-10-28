import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import battery as bt
from predictor import DATA_DIR, latlon_to_grid, load_model, predict_location, tomorrow_forecast

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


PRICE_FILE = DATA_DIR / "jeju_smp_hourly.csv"
EPSIS_URL = "https://epsis.kpx.or.kr/epsisnew/selectEkmaSmpShdChart.do?menuId=040202"
ACTION_ICON = {"충전": "🔋⬆️", "방전": "🔋⬇️", "대기": "⏸️"}
TOL_HELP = ("약속(급전지시)보다 이 범위를 넘게 더 보내면 임밸런스 페널티가 붙어요. "
            "전력시장운영규칙 [별표 33] 제주 시범사업 정산기준 3.가.(2)(라) 기준 8% (2025년, 2024년까지는 12%). "
            "시기에 따라, 또는 입찰제도가 육지로 확대되면서 기준이 달라질 수 있으니 최신 규칙에 맞게 설정하세요.")


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
    st.session_state.loc = (35.2772, 126.5120)   # 처음 위치: 전남 영광

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
    key = f"_{preset_name}"   # 유형을 바꾸면 입력칸 기본값도 바뀌도록
    energy = st.number_input("배터리 용량 (kWh)", 0.0, 400_000.0, float(round(capacity * ps["duration_h"])), 100.0,
                             key="e" + key)
    power = st.number_input("최대 충·방전 출력 (kW)", 0.0, 200_000.0,
                            float(round(energy / ps["duration_h"])) if energy else 0.0, 100.0, key="p" + key)
    efficiency = st.slider("왕복 효율", 0.70, 0.98, ps["efficiency"], 0.01, key="ef" + key)
    soh = st.slider("현재 건강도 SOH (%)", 50, 100, int(ps["soh"] * 100), 1, key="soh" + key,
                    help="새 배터리는 100%. 오래 써서 용량이 줄었으면 낮춰 주세요.") / 100
    capex = st.number_input("배터리 구입·설치비 (원/kWh)", 50_000, 2_000_000, ps["capex"], 10_000, key="c" + key)
    cycle_life = st.number_input("수명 사이클 (회, 80% DoD 기준)", 500, 20_000, ps["cycle_life"], 500, key="cl" + key)
    soc_range = st.slider("사용할 충전량 범위 (%)", 0, 100, (10, 90), 5, key="soc" + key)

    st.divider()
    st.header("전기 가격")
    st.link_button("📈 전력거래소 시간별 SMP 바로가기", EPSIS_URL, use_container_width=True)
    st.caption("구분(제주/육지)과 기간을 고르고 **조회 → CSV**로 받아서 아래에 올려 주세요. 최근 며칠치면 충분해요.")
    up = st.file_uploader("EPSIS '시간별 SMP' CSV", type=["csv"])
    prices = None
    if up is not None:
        prices = load_price_file(up.getvalue())
    elif PRICE_FILE.exists():
        prices = load_price_file(PRICE_FILE.read_bytes())
    if prices is not None and len(prices):
        st.caption(f"가격 데이터: {prices['date'].min():%Y-%m-%d} ~ {prices['date'].max():%Y-%m-%d}")
    else:
        st.caption("⚠️ 가격 데이터가 없어서 모든 시간을 100원으로 가정해요.")

bcfg = bt.BatteryConfig(energy_kwh=energy, power_kw=power, soh=soh, efficiency=efficiency,
                        capex_won_per_kwh=capex, cycle_life=cycle_life,
                        soc_min=soc_range[0] / 100, soc_max=soc_range[1] / 100)


tab1, tab4, tab3, tab2 = st.tabs(["☀️ 내일 발전량 예측", "⚡ 실시간 운영 판단", "🔋 2025년 성과 검증", "📊 모델 성능"])

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
        st.subheader("🧾 내일 추천 입찰표 (AI 예측 + 배터리)")
        st.caption(
            "AI가 예측한 발전량에 배터리 운전 계획을 얹어서, 시간별로 얼마나 팔겠다고 입찰할지 정해요. "
            "싼 시간에는 덜 팔고 배터리에 담았다가, 비싼 시간에 꺼내 파는 계획이에요. 배터리가 닳는 비용까지 빼고 계산해요."
        )
        if prices is not None and len(prices):
            exp_price = bt.expected_prices(prices, days=7)
            price_note = f"예상 가격 = 올린 가격 데이터의 최근 7일 시간대별 평균 ({prices['date'].max():%m/%d}까지)"
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
            help="계획에 쓰지 않고 남겨 두는 양이에요. AI는 흐린 날 실제보다 많게 예측하는 경향이 있어서, "
                 "모자랄 때 당일 메우려고 남겨 둬요. 비·높은 습도 예보가 있으면 자동으로 20%로 올려요.",
        ) / 100
        if rainy:
            st.info("🌧️ 내일 낮에 비나 높은 습도가 예보돼 있어요. AI가 과대예측할 수 있어서 예비 충전량을 늘렸어요.")

        plan, psum = bt.plan_day(res["pred_kwh"].values, exp_price.values, bcfg, capacity, soc_start, reserve)
        plan.index = range(24)
        plan["soc_before"] = np.r_[soc_start, plan["soc"].values[:-1]]
        st.session_state["bid_plan"] = plan

        k1, k2, k3 = st.columns(3)
        k1.metric("배터리 없이 그대로 입찰", f"{psum['배터리 없이 예상 수입(원)'] / 10_000:,.1f}만 원")
        k2.metric("배터리 포함 입찰 (감가 차감)",
                  f"{(psum['배터리 포함 예상 수입(원)'] - psum['배터리 감가비용(원)']) / 10_000:,.1f}만 원",
                  f"{psum['순이득(원)'] / 10_000:+,.1f}만 원")
        k3.metric("배터리 감가비용", f"{psum['배터리 감가비용(원)'] / 10_000:,.1f}만 원")
        st.caption(price_note)

        view = pd.DataFrame({
            "AI 예측 발전량": plan["gen_kwh"].values,
            "추천 입찰량": plan["bid_kwh"].values,
        }, index=[f"{h:02d}시" for h in range(24)])
        st.line_chart(view, y_label="kWh")

        with st.expander("시간별 입찰표 보기 · 받기"):
            tbl = pd.DataFrame({
                "시간": [f"{h:02d}시" for h in range(24)],
                "AI 예측 발전량(kWh)": plan["gen_kwh"].round(0),
                "예상 가격(원/kWh)": plan["price"].round(1),
                "배터리 충전(kWh)": plan["charge_kwh"].round(0),
                "배터리 방전(kWh)": plan["discharge_kwh"].round(0),
                "추천 입찰량(kWh)": plan["bid_kwh"].round(0),
                "배터리 충전량 SOC(%)": (plan["soc"] * 100).round(0),
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True)
            st.download_button("입찰표 CSV로 받기", tbl.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"추천입찰표_{day}.csv", mime="text/csv")
        st.caption("이 입찰표는 '⚡ 실시간 운영 판단' 탭의 약속량으로 자동 연결돼요.")

# ---------- 탭 2: 모델 성능 ----------
with tab2:
    m = get_metrics()
    st.subheader("지역별 오차율 (2025년 시험기간)")
    st.caption("용량기준 오차율 = |예측 − 실제| ÷ 설비용량. 낮을수록 좋아요. 예측제도 정산 기준은 8% 이하예요.")
    err = m.pivot(index="지역", columns="모델", values="용량기준 오차율(%)")
    err.columns = ["모델 A (관측 기반)", "모델 B (예보 대응·웹 사용)"]
    st.bar_chart(err, stack=False)
    st.dataframe(m, use_container_width=True, hide_index=True)


# ---------- 탭 3: 배터리 거래 시뮬레이션 (2025년 실제 데이터로 검증) ----------
HIST_FILE = DATA_DIR / "site_hourly_2025.csv"


@st.cache_data
def get_history():
    return pd.read_csv(HIST_FILE, encoding="utf-8-sig", parse_dates=["datetime"])


@st.cache_data(show_spinner="배터리 운전 시뮬레이션 중...")
def run_sim(df, b_kwargs, m_kwargs):
    return bt.compare(df, bt.BatteryConfig(**b_kwargs), bt.MarketConfig(**m_kwargs))


def won(x):
    return f"{x / 10_000:,.0f}만 원"


with tab3:
    st.subheader("이 엔진을 2025년에 썼다면? (성과 검증)")
    st.caption(
        "2025년 실제 발전량과 AI 예측값으로 세 가지 운전 방식을 비교해요. "
        "**단순 운전**은 오차가 나면 무조건 배터리를 쓰고, **감가 고려(Voltics)**는 "
        "'아끼는 페널티 > 배터리가 닳는 비용'일 때만, 그리고 배터리가 빨리 닳는 충전 구간을 피해서 써요."
    )

    if not HIST_FILE.exists():
        st.warning(
            "시뮬레이션 데이터(site_hourly_2025.csv)가 아직 없어요. "
            "test_predictions.csv를 줄인 파일을 data/output 폴더에 올려 주세요."
        )
    else:
        hist = get_history()

        st.markdown("#### ① 비교 조건  ·  배터리 사양은 왼쪽 '내 배터리'에서 바꿔요")
        c1, c2, c3 = st.columns(3)
        sim_site = c1.selectbox("가장 가까운 지역 (날씨 패턴)", sorted(hist["site"].unique()), key="sim_site")
        my_cap = capacity
        c2.metric("내 발전소 설비용량 (왼쪽에서 변경)", f"{capacity:,.0f} kW")
        h = hist[hist["site"] == sim_site]
        dmin, dmax = h["datetime"].min().date(), h["datetime"].max().date()
        period = c3.date_input("검증 기간", (dmin, dmax), min_value=dmin, max_value=dmax)

        with st.expander("⚙️ 입찰·정산 가정 (경영 파트 정산식이 나오면 여기 값을 맞추세요)"):
            e1, e2, e3, e4 = st.columns(4)
            bid_factor = e1.slider(
                "입찰 보정계수", 0.6, 1.2, 1.0, 0.05,
                help="입찰량 = 예측 × 계수. 2025년 데이터에서는 1.0(예측 그대로)이 가장 유리했어요.",
            )
            tolerance = e2.slider("페널티 없는 오차 범위 (설비용량 대비 %)", 0.0, 20.0, 8.0, 0.5, help=TOL_HELP)
            penalty_ratio = e3.slider("페널티 단가 (SMP의 몇 배)", 0.0, 3.0, 1.0, 0.1)
            stress = e4.slider("고·저 충전 구간 열화 배수", 1.0, 4.0, 2.0, 0.5,
                               help="충전량 80% 이상·20% 이하에서 쓰면 이만큼 더 빨리 닳는다고 가정")

        st.caption(
            f"→ 이 배터리로 1kWh를 움직일 때 닳는 비용 ≈ **{capex / (2 * cycle_life * 0.8):,.1f}원**. "
            f"실제로 쓸 수 있는 용량 = {energy:,.0f} × SOH {soh:.0%} = **{energy * soh:,.0f} kWh**. "
            "아끼는 페널티가 닳는 비용보다 클 때만 Voltics가 배터리를 씁니다."
        )

        if isinstance(period, tuple) and len(period) == 2:
            start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1]) + pd.Timedelta(days=1)
            df = bt.scale_to_plant(
                h[(h["datetime"] >= start) & (h["datetime"] < end)][["datetime", "cap_kw", "gen_kwh", "pred_kwh", "smp"]],
                my_cap,
            )
            b_kwargs = dict(energy_kwh=energy, power_kw=power, soh=soh, capex_won_per_kwh=capex,
                            cycle_life=cycle_life, efficiency=efficiency, stress_factor=stress,
                            soc_min=soc_range[0] / 100, soc_max=soc_range[1] / 100,
                            soc_init=(soc_range[0] + soc_range[1]) / 200)
            m_kwargs = dict(bid_factor=bid_factor, tolerance_pct=tolerance, penalty_ratio=penalty_ratio)
            table, runs = run_sim(df, b_kwargs, m_kwargs)

            base = table.loc[bt.STRATEGIES["none"], "순수익(원)"]
            naive = table.loc[bt.STRATEGIES["naive"], "순수익(원)"]
            aware = table.loc[bt.STRATEGIES["aware"], "순수익(원)"]

            st.markdown("#### ③ 결과")
            st.markdown(f"**내 발전소 {my_cap:,.0f} kW ({sim_site} 날씨)** · 배터리 {energy:,.0f} kWh / {power:,.0f} kW "
                        f"· SOH {soh:.0%} · {period[0]} ~ {period[1]}")
            k1, k2, k3 = st.columns(3)
            k1.metric("배터리 없음 순수익", won(base))
            k2.metric("단순 운전 순수익", won(naive), f"{won(naive - base)} vs 배터리 없음")
            k3.metric("Voltics 순수익", won(aware), f"{won(aware - base)} vs 배터리 없음")
            if aware > naive:
                st.success(f"수명 감가를 고려하면 단순 운전보다 **{won(aware - naive)}** 더 벌어요.")

            st.markdown("##### 어디서 차이가 나나")
            show = table[["페널티(원)", "배터리 감가비용(원)", "순수익(원)"]] / 10_000
            show.columns = ["페널티(만 원)", "배터리 감가비용(만 원)", "순수익(만 원)"]
            st.bar_chart(show[["페널티(만 원)", "배터리 감가비용(만 원)"]], stack=False)

            nice = table.copy()
            for c in ["판매수익(원)", "페널티(원)", "배터리 감가비용(원)", "순수익(원)"]:
                nice[c] = nice[c].map(lambda v: f"{v:,.0f}")
            nice["등가 사이클(회)"] = table["등가 사이클(회)"].map(lambda v: f"{v:,.1f}")
            nice["수명 소모(%p)"] = table["수명 소모(%p)"].map(lambda v: f"{v:.2f}")
            st.dataframe(nice, use_container_width=True)

            st.markdown("##### 누적 순수익 차이 (배터리 없음 대비)")
            daily = pd.DataFrame({
                bt.STRATEGIES[k]: r.set_index("datetime")["net"].resample("D").sum()
                for k, r in runs.items()
            })
            gain = daily.sub(daily[bt.STRATEGIES["none"]], axis=0).drop(columns=bt.STRATEGIES["none"])
            st.line_chart(gain.cumsum() / 10_000, y_label="만 원")

            st.markdown("##### 하루 들여다보기")
            days = sorted(df["datetime"].dt.date.unique())
            pick = st.select_slider("날짜", options=days, value=days[min(len(days) - 1, len(days) // 2)])
            one = runs["aware"][runs["aware"]["datetime"].dt.date == pick].set_index(
                runs["aware"]["datetime"][runs["aware"]["datetime"].dt.date == pick].dt.hour
            )
            g1, g2 = st.columns(2)
            with g1:
                st.caption("입찰량 vs 실제 vs 배터리 보정 후 (kWh)")
                st.line_chart(one[["bid_kwh", "gen_kwh", "delivered_kwh"]].rename(columns={
                    "bid_kwh": "입찰량", "gen_kwh": "실제 발전", "delivered_kwh": "배터리 보정 후"}))
            with g2:
                st.caption("배터리 충전량 SOC (%)")
                st.area_chart(one[["soc"]].rename(columns={"soc": "SOC(%)"}) * 100)

            st.info("페널티 규칙은 임시 가정이에요. 경영 파트의 KPX 정산식이 나오면 battery.py의 penalty_cost() 함수만 바꾸면 돼요.")

# ---------- 탭 4: 실시간 운영 판단 ----------

with tab4:
    st.subheader("지금 이 시간, 배터리를 어떻게 할까?")
    st.caption(
        "하루 전 약속한 양(낙찰량)과 지금 예상되는 발전량이 다를 때, "
        "**충전 · 방전 · 대기** 중 돈이 가장 많이 남는 쪽을 배터리가 닳는 비용까지 빼고 골라 줘요."
    )

    # --- ① 오늘 가격표: EPSIS 시간별 SMP(하루전 가격) + 지금 실시간 가격 ---
    st.markdown("#### ① 오늘 가격표")
    st.caption("가격 데이터는 왼쪽 사이드바 '전기 가격'에서 올려요.")

    now_kst = pd.Timestamp.now(tz="Asia/Seoul")
    p1, p2 = st.columns(2)
    if prices is not None and len(prices):
        dates = sorted(prices["date"].dt.date.unique())
        today = now_kst.date()
        day_pick = p1.selectbox("날짜", dates[::-1], index=0 if today not in dates else dates[::-1].index(today))
        day_prices = prices[prices["date"].dt.date == day_pick].set_index("hour")["price"].reindex(range(24))
    else:
        p1.info("가격 파일이 없어서 빈 표로 시작해요. 아래 표에 직접 넣어도 돼요.")
        day_prices = pd.Series([100.0] * 24, index=range(24))
    hour = p2.selectbox("판단할 시간", list(range(24)), index=int(now_kst.hour),
                        format_func=lambda h: f"{h:02d}시 ~ {h + 1:02d}시")

    with st.expander("가격표 보기·고치기 (원/kWh)"):
        edited = st.data_editor(
            pd.DataFrame({"시간": [f"{h:02d}시" for h in range(24)], "가격": day_prices.fillna(0).values}),
            hide_index=True, use_container_width=True, key=f"pt_{hour}",
        )
        day_prices = pd.Series(edited["가격"].astype(float).values, index=range(24))

    price_now = st.number_input(
        f"⚡ 지금({hour:02d}시) 실시간 가격 (원/kWh)", -100.0, 1000.0, float(day_prices[hour]), 1.0,
        help="실제 실시간 가격을 알면 넣어 주세요. 비워 두면 가격표(하루전 가격) 값을 그대로 써요.",
    )
    future = day_prices[day_prices.index > hour]

    # --- ② 이번 시간 상황 ---
    st.markdown("#### ② 이번 시간 상황")
    s1, s2, s3 = st.columns(3)
    plan = st.session_state.get("bid_plan")
    if plan is not None:
        st.caption("☀️ 탭 1의 **AI 추천 입찰표**에서 약속량·예상 발전량·충전량을 가져왔어요. "
                   "실제 상황(예: 갑자기 흐려짐)에 맞게 '지금 예상되는 발전량'을 고쳐 보세요.")
        d_prom, d_exp, d_soc = (float(round(plan.loc[hour, "bid_kwh"])), float(round(plan.loc[hour, "gen_kwh"])),
                                int(round(plan.loc[hour, "soc_before"] * 100)))
    else:
        st.caption("탭 1에서 내일 예측을 먼저 하면, AI 추천 입찰표 값이 여기에 자동으로 들어와요.")
        d_prom, d_exp, d_soc = float(round(capacity * 0.6)), float(round(capacity * 0.45)), 50
    promised = s1.number_input("하루 전 약속한 양 (kWh)", 0.0, 1_000_000.0, d_prom, 10.0, key=f"prom_{hour}_{d_prom}",
                               help="이 시간에 팔겠다고 낙찰받은 양 (탭 1 추천 입찰표)")
    expected = s2.number_input("지금 예상되는 실제 발전량 (kWh)", 0.0, 1_000_000.0, d_exp, 10.0, key=f"exp_{hour}_{d_exp}",
                               help="AI 예측값이 기본값이에요. 현장 계측값이나 최신 예보로 다시 예측한 값을 넣으세요.")
    soc_now = s3.slider("지금 배터리 충전량 SOC (%)", 0, 100, d_soc, 1, key=f"soc_{hour}_{d_soc}") / 100
    tol = st.slider("페널티 없는 초과 범위 (설비용량 대비 %)", 0.0, 20.0, 8.0, 0.5, help=TOL_HELP)

    best, opts, info = bt.decide_hour(promised, expected, price_now, future.values, capacity,
                                      bcfg, soc_now, bt.MarketConfig(tolerance_pct=tol))

    # --- ③ 추천 ---
    st.markdown("#### ③ 추천")
    gap = info["dev"]
    gap_txt = f"약속보다 **{abs(gap):,.0f} kWh 모자라요**" if gap < 0 else (
        f"약속보다 **{gap:,.0f} kWh 남아요**" if gap > 0 else "약속과 **딱 맞아요**")
    st.markdown(
        f"{gap_txt} · 지금 가격 **{price_now:,.0f}원** · 오늘 남은 시간 최고가 **{info['future_best']:,.0f}원** · "
        f"배터리 1kWh 쓸 때 닳는 비용 **{info['c_deg']:,.1f}원**"
    )
    base = opts.loc[opts["action"] == "대기", "total"].iloc[0]
    msg = f"## {ACTION_ICON[best.action]} {best.action}" + (f" {best.kwh:,.0f} kWh" if best.kwh else "")
    st.success(msg + f"\n\n대기보다 **{best.total - base:,.0f}원** 이득 · {best.why}")
    if gap < 0 and best.action != "방전":
        st.caption("💡 모자라지만 지금 배터리로 메우지 않는 게 나아요. 모자란 만큼은 실시간 가격으로 정산되고, "
                   "배터리 전기는 더 비싼 시간에 쓰는 편이 이득이에요.")

    show = opts[["action", "kwh", "now", "later", "deg", "total"]].copy()
    show.columns = ["선택지", "양(kWh)", "이번 시간 정산(원)", "나중 시간 가치(원)", "배터리 감가(원)", "합계(원)"]
    st.dataframe(show.style.format({c: "{:,.0f}" for c in show.columns[1:]}),
                 use_container_width=True, hide_index=True)
    st.caption("이번 시간 정산 = (실제로 보낸 양 − 약속량) × 지금 가격 − 초과 페널티. "
               "약속량만큼의 수익은 하루 전에 이미 확정돼서 여기선 '차이'만 계산해요.")

    chart = pd.DataFrame({"가격(원/kWh)": day_prices.values}, index=[f"{h:02d}시" for h in range(24)])
    chart.loc[f"{hour:02d}시", "가격(원/kWh)"] = price_now
    st.caption("오늘 가격 흐름 (지금 시간은 실시간 가격으로 바꿔서 표시)")
    st.bar_chart(chart)

    st.info(
        "제주 시범사업 규칙상 실시간시장 입찰은 1시간 단위이고 **거래시간 75분 전에 마감**돼요. "
        "그래서 실제 운영에서는 '75분 뒤 시간'의 배터리 계획을 미리 정해 변경입찰에 반영하는 식으로 써요."
    )
