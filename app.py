import pandas as pd
import streamlit as st

from predictor import DATA_DIR, GRID, load_model, predict_day, sample_forecast, tomorrow_forecast

st.set_page_config(page_title="VPP 발전량 예측", page_icon="☀️", layout="wide")


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

st.title("☀️ VPP 태양광 발전량 예측 대시보드")
st.caption("발전소 용량과 지역을 고르면, 기상청 단기예보로 내일 시간대별 발전량을 예측합니다.")

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

tab1, tab2, tab3 = st.tabs(["내일 발전량 예측", "모델 성능", "입찰 추천"])

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

# ---------- 탭 3: 입찰 추천 (경영 파트 정산식이 나오면 채우기) ----------
with tab3:
    st.info("경영 파트의 정산식·입찰 전략이 완성되면 이곳에 추천 입찰량과 예상 수익을 표시할 예정이에요.")
