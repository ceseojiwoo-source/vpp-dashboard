import pandas as pd
import streamlit as st

import battery as bt
from predictor import DATA_DIR, GRID, load_model, predict_day, sample_forecast, tomorrow_forecast

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


def get_key():
    try:
        return st.secrets["KMA_KEY"]
    except Exception:
        return ""


model, meta = get_model()

st.title("⚡ Voltics")
st.markdown("**배터리 수명 감가 고려형 AI 전력 거래 엔진** · 재생에너지 입찰제도 대응")
st.caption("① AI가 내일 발전량을 예측하고 → ② 예측 오차는 배터리로 메우되, 배터리가 닳는 비용까지 따져서 가장 이득인 운전을 고릅니다.")

# ---------- 왼쪽 입력창 ----------
with st.sidebar:
    st.header("발전소 정보")
    site = st.selectbox("지역", meta["sites"])
    capacity = st.number_input("설비용량 (kW)", min_value=1.0, value=1000.0, step=100.0)

    st.divider()
    st.header("날씨 데이터")
    mode = st.radio("예보 가져오기", ["기상청 실시간 예보", "예시 날씨 (테스트용)"])
    if mode == "기상청 실시간 예보":
        nx0, ny0 = GRID.get(site, (60, 127))
        nx = st.number_input("격자 X (nx)", value=nx0, step=1)
        ny = st.number_input("격자 Y (ny)", value=ny0, step=1)
        if site not in GRID:
            st.caption("⚠️ 내륙 지역 격자는 아직 정해지지 않아 서울 값이 들어가 있어요. 팀원과 정한 값으로 바꿔 주세요.")

tab1, tab3, tab2 = st.tabs(["☀️ 내일 발전량 예측", "🔋 배터리 거래 시뮬레이션", "📊 모델 성능"])

# ---------- 탭 1: 내일 예측 ----------
with tab1:
    forecast, day = None, None
    if mode == "기상청 실시간 예보":
        key = get_key()
        if not key:
            st.warning("기상청 API 키가 없어요. Streamlit 설정의 Secrets에 KMA_KEY를 넣거나, 왼쪽에서 '예시 날씨'를 고르세요.")
        else:
            try:
                forecast, day = get_forecast(key, nx, ny)
            except Exception as e:
                st.error(f"예보를 받아오지 못했어요: {e}")
    else:
        forecast, day = sample_forecast()
        st.info("테스트용 예시 날씨(맑은 날 가정)로 예측한 결과예요.")

    if forecast is not None and len(forecast) == 0:
        st.error("내일 예보가 비어 있어요. 잠시 후 다시 시도해 주세요.")
    elif forecast is not None:
        res = predict_day(model, meta, capacity, forecast, site)
        total = res["pred_kwh"].sum()
        peak = res.loc[res["pred_kwh"].idxmax()]

        st.subheader(f"{site} · {capacity:,.0f} kW · {day} 예측")
        c1, c2, c3 = st.columns(3)
        c1.metric("하루 총 예상 발전량", f"{total:,.0f} kWh")
        c2.metric("가장 많이 나오는 시간", f"{peak['datetime'].hour}시 · {peak['pred_kwh']:,.0f} kWh")
        c3.metric("하루 이용률", f"{total / (capacity * 24) * 100:.1f} %")

        chart = res.assign(시각=res["datetime"].dt.hour).set_index("시각")[["pred_kwh"]]
        st.bar_chart(chart.rename(columns={"pred_kwh": "예측 발전량 (kWh)"}))

        # 이 지역 모델 정확도 안내
        m = get_metrics()
        row = m[(m["지역"] == site) & (m["모델"] == "B_예보대응")]
        if not row.empty:
            err = row["용량기준 오차율(%)"].iloc[0]
            st.caption(
                f"참고: 이 지역에서 모델의 평균 오차는 설비용량 대비 약 {err:.1f}%예요. "
                "흐린 날에는 실제보다 많게 예측하는 경향이 있어요."
            )

        with st.expander("시간별 자세히 보기"):
            table = forecast.merge(res, on="datetime")
            table.columns = ["시각", "기온(℃)", "습도(%)", "풍속(m/s)", "강수(mm)", "예측 발전량(kWh)"]
            st.dataframe(table.round(1), use_container_width=True, hide_index=True)
            st.download_button(
                "CSV로 받기", table.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"예측_{site}_{day}.csv", mime="text/csv",
            )

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
    st.subheader("배터리 수명 감가를 고려하면 얼마나 더 버는가?")
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

        c1, c2 = st.columns(2)
        with c1:
            sim_site = st.selectbox("지역", sorted(hist["site"].unique()), key="sim_site")
            h = hist[hist["site"] == sim_site]
            dmin, dmax = h["datetime"].min().date(), h["datetime"].max().date()
            period = st.date_input("기간", (dmin, dmax), min_value=dmin, max_value=dmax)
        with c2:
            bid_factor = st.slider(
                "입찰 보정계수 (입찰량 = 예측 × 계수)", 0.6, 1.2, 1.0, 0.05,
                help="AI 모델이 흐린 날 많게 예측하는 경향이 있어서, 1보다 조금 낮추면 페널티가 줄 수 있어요.",
            )
            hours = st.slider("배터리 크기 (발전소 용량 × 몇 시간)", 0.0, 4.0, 1.0, 0.25,
                              help="1.0이면 1,000kW 발전소에 1,000kWh 배터리")

        with st.expander("⚙️ 배터리·정산 가정 바꾸기 (경영 파트 정산식이 나오면 여기 값을 맞추세요)"):
            e1, e2, e3 = st.columns(3)
            capex = e1.number_input("배터리 설치비 (원/kWh)", 50_000, 2_000_000, 400_000, 50_000)
            cycle_life = e2.number_input("수명 사이클 (회)", 1_000, 20_000, 6_000, 500)
            efficiency = e3.slider("왕복 효율", 0.70, 0.98, 0.90, 0.01)
            e4, e5, e6 = st.columns(3)
            tolerance = e4.slider("페널티 없는 오차 범위 (설비용량 대비 %)", 0.0, 20.0, 8.0, 0.5)
            penalty_ratio = e5.slider("페널티 단가 (SMP의 몇 배)", 0.0, 3.0, 1.0, 0.1)
            stress = e6.slider("고·저 충전 구간 열화 배수", 1.0, 4.0, 2.0, 0.5,
                               help="충전량 80% 이상·20% 이하에서 쓰면 이만큼 더 빨리 닳는다고 가정")
            st.caption(
                f"→ 배터리로 1kWh를 움직일 때 닳는 비용 ≈ **{capex / (2 * cycle_life * 0.8):,.1f}원**. "
                "이 값보다 아끼는 페널티가 클 때만 Voltics가 배터리를 씁니다."
            )

        if isinstance(period, tuple) and len(period) == 2:
            start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1]) + pd.Timedelta(days=1)
            df = h[(h["datetime"] >= start) & (h["datetime"] < end)][
                ["datetime", "cap_kw", "gen_kwh", "pred_kwh", "smp"]
            ]
            b_kwargs = dict(hours=hours, capex_won_per_kwh=capex, cycle_life=cycle_life,
                            efficiency=efficiency, stress_factor=stress)
            m_kwargs = dict(bid_factor=bid_factor, tolerance_pct=tolerance, penalty_ratio=penalty_ratio)
            table, runs = run_sim(df, b_kwargs, m_kwargs)

            base = table.loc[bt.STRATEGIES["none"], "순수익(원)"]
            naive = table.loc[bt.STRATEGIES["naive"], "순수익(원)"]
            aware = table.loc[bt.STRATEGIES["aware"], "순수익(원)"]

            st.markdown(f"**{sim_site}** · 설비 {df['cap_kw'].iloc[0]:,.0f} kW · "
                        f"배터리 {df['cap_kw'].iloc[0] * hours:,.0f} kWh · {period[0]} ~ {period[1]}")
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
