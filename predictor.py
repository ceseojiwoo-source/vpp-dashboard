"""
팀원 노트북(VPP_발전량예측.ipynb)의 예측 함수를 웹에서 쓸 수 있게 옮긴 파일.
노트북 10~11번 셀의 time_features / predict_day / kma_forecast 와 같은 계산을 합니다.
"""
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote

import joblib
import numpy as np
import pandas as pd
import requests

# 모델 파일 위치: data/output/ 이 있으면 거기서, 없으면 data/ 에서 찾습니다.
_BASE = Path(__file__).parent / "data"
DATA_DIR = _BASE / "output" if (_BASE / "output" / "model_B.pkl").exists() else _BASE
KST = timezone(timedelta(hours=9))

# 기상청 단기예보 격자 좌표 (nx, ny). 내륙은 팀에서 대표 지점을 정하면 추가하세요.
GRID = {
    "제주#1": (53, 38),
    "한빛#1": (52, 77),
}


# ---------------------------------------------------------------------------
# 1) 모델 불러오기
# ---------------------------------------------------------------------------
def load_model():
    model = joblib.load(DATA_DIR / "model_B.pkl")
    with open(DATA_DIR / "model_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


# ---------------------------------------------------------------------------
# 2) 시간/계절/태양고도 변수 만들기 (노트북 6번 셀과 동일)
# ---------------------------------------------------------------------------
def time_features(t, lat):
    t = pd.Series(pd.to_datetime(t)).reset_index(drop=True)
    lat_r = np.radians(np.asarray(lat, dtype=float))
    hr, doy = t.dt.hour.values, t.dt.dayofyear.values
    out = pd.DataFrame({"hour": hr, "month": t.dt.month.values, "doy": doy})
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    out["doy_sin"], out["doy_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + doy) / 365)
    ha = np.radians(15 * (hr + 0.5 - 12))  # 구간 중앙 시각 기준
    sin_alt = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(ha)
    out["sun_alt"] = np.clip(sin_alt, 0, None)  # 밤 = 0
    out["lat"] = np.degrees(lat_r)
    return out


# ---------------------------------------------------------------------------
# 3) 발전량 예측 (노트북 10번 셀의 predict_day)
#    forecast_df 컬럼: datetime, temp, rain, wind, hum
# ---------------------------------------------------------------------------
def predict_day(model, meta, capacity_kw, forecast_df, site):
    if site not in meta["sites"]:
        raise ValueError(f"site는 {meta['sites']} 중 하나여야 합니다")
    f = forecast_df.copy().reset_index(drop=True)
    f["datetime"] = pd.to_datetime(f["datetime"])
    region = re.match(r"^[가-힣]+", site).group(0)  # '제주#1' -> '제주'
    lat = meta["lat_by_region"].get(region, 36.0)
    X = pd.concat(
        [
            time_features(f["datetime"], np.full(len(f), lat)),
            pd.DataFrame({f"site_{x}": int(x == site) for x in meta["sites"]}, index=f.index),
            f[["temp", "rain", "wind", "hum"]],
        ],
        axis=1,
    )[meta["features_B"]]
    cf = np.clip(model.predict(X), 0, 1)  # 이용률
    cf[X["sun_alt"].values == 0] = 0      # 해 없으면 0
    return pd.DataFrame({"datetime": f["datetime"], "pred_kwh": cf * capacity_kw})


# ---------------------------------------------------------------------------
# 4) 기상청 단기예보 받아오기 (노트북 11번 셀의 kma_forecast)
# ---------------------------------------------------------------------------
def _pcp(v):
    v = str(v)
    if "없음" in v or v == "nan":
        return 0.0
    if "미만" in v:
        return 0.5
    n = re.findall(r"[\d.]+", v)
    return float(n[0]) if n else 0.0


def kma_forecast(key, nx, ny):
    now = datetime.now(KST) - timedelta(minutes=15)  # 발표 직후엔 아직 자료가 없을 수 있어 15분 여유
    base_hours = [2, 5, 8, 11, 14, 17, 20, 23]
    past = [h for h in base_hours if h <= now.hour]
    if past:
        bh, base_day = past[-1], now
    else:
        bh, base_day = 23, now - timedelta(days=1)

    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {
        "serviceKey": unquote(key.strip()),  # Encoding/Decoding 키 둘 다 받을 수 있게
        "pageNo": 1, "numOfRows": 1500, "dataType": "JSON",
        "base_date": base_day.strftime("%Y%m%d"), "base_time": f"{bh:02d}00",
        "nx": int(nx), "ny": int(ny),
    }
    r = requests.get(url, params=params, timeout=20)
    try:
        body = r.json()
    except ValueError:
        raise RuntimeError("기상청 응답을 읽지 못했어요. API 키가 맞는지 확인하세요.\n" + r.text[:300])
    header = body["response"]["header"]
    if header.get("resultCode") != "00":
        raise RuntimeError(f"기상청 오류: {header.get('resultMsg')}")

    items = body["response"]["body"]["items"]["item"]
    t = (
        pd.DataFrame(items)
        .pivot_table(index=["fcstDate", "fcstTime"], columns="category", values="fcstValue", aggfunc="first")
        .reset_index()
    )
    return pd.DataFrame({
        "datetime": pd.to_datetime(t["fcstDate"] + t["fcstTime"], format="%Y%m%d%H%M"),
        "temp": pd.to_numeric(t["TMP"], errors="coerce"),
        "hum": pd.to_numeric(t["REH"], errors="coerce"),
        "wind": pd.to_numeric(t["WSD"], errors="coerce"),
        "rain": t["PCP"].map(_pcp),
    })


def tomorrow_forecast(key, nx, ny):
    """내일 24시간 예보만 잘라서 돌려줍니다. (예보표, 내일 날짜)"""
    fc = kma_forecast(key, nx, ny)
    tmr = (datetime.now(KST) + timedelta(days=1)).date()
    return fc[fc["datetime"].dt.date == tmr].reset_index(drop=True), tmr


# ---------------------------------------------------------------------------
# 5) API 키 없이 화면을 테스트할 때 쓰는 예시 날씨 (맑은 날 가정)
# ---------------------------------------------------------------------------
def sample_forecast():
    tmr = (datetime.now(KST) + timedelta(days=1)).date()
    hours = np.arange(24)
    return pd.DataFrame({
        "datetime": pd.to_datetime(str(tmr)) + pd.to_timedelta(hours, unit="h"),
        "temp": 15 + 6 * np.sin((hours - 9) / 24 * 2 * np.pi),
        "hum": 70 - 25 * np.sin((hours - 9) / 24 * 2 * np.pi),
        "wind": 2.0,
        "rain": 0.0,
    }), tmr


# ---------------------------------------------------------------------------
# 6) 위도·경도 → 기상청 격자(nx, ny) 변환 (기상청 공식 Lambert 투영식)
#    기상청은 전국을 5km 네모칸으로 나누고 칸마다 번호(nx, ny)를 붙여 예보한다.
# ---------------------------------------------------------------------------
def latlon_to_grid(lat, lon):
    import math
    RE, GRID_KM, SLAT1, SLAT2, OLON, OLAT, XO, YO = 6371.00877, 5.0, 30.0, 60.0, 126.0, 38.0, 43, 136
    D = math.pi / 180.0
    re = RE / GRID_KM
    slat1, slat2, olon, olat = SLAT1 * D, SLAT2 * D, OLON * D, OLAT * D
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(
        math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5))
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5) ** sn * math.cos(slat1) / sn
    ro = re * sf / math.tan(math.pi * 0.25 + olat * 0.5) ** sn
    ra = re * sf / math.tan(math.pi * 0.25 + lat * D * 0.5) ** sn
    theta = lon * D - olon
    theta = (theta + 2 * math.pi) if theta < -math.pi else (theta - 2 * math.pi) if theta > math.pi else theta
    theta *= sn
    return int(ra * math.sin(theta) + XO + 0.5), int(ro - ra * math.cos(theta) + YO + 0.5)


# ---------------------------------------------------------------------------
# 7) 아무 위치나 예측하기
#    모델은 5개 학습 지역 중 하나를 골라야 한다(지역마다 발전소 특성이 조금씩 다름).
#    - 제주도 안이면 제주#1
#    - 육지면 육지 4곳(내륙#1~3, 한빛#1)으로 각각 예측해서 평균
#      (내륙 지역의 실제 위치가 확인되면 '가장 가까운 지역'으로 바꾸면 됨)
#    태양 고도는 고른 위치의 위도로 정확히 계산한다.
# ---------------------------------------------------------------------------
def is_jeju(lat, lon):
    return 33.0 <= lat <= 33.7 and 126.0 <= lon <= 127.0


def reference_sites(lat, lon, meta):
    if is_jeju(lat, lon):
        return [s for s in meta["sites"] if s.startswith("제주")]
    return [s for s in meta["sites"] if not s.startswith("제주")]


def predict_location(model, meta, capacity_kw, forecast_df, lat, lon):
    sites = reference_sites(lat, lon, meta)
    f = forecast_df.copy().reset_index(drop=True)
    f["datetime"] = pd.to_datetime(f["datetime"])
    tf = time_features(f["datetime"], np.full(len(f), lat))
    preds = []
    for site in sites:
        X = pd.concat(
            [tf, pd.DataFrame({f"site_{x}": int(x == site) for x in meta["sites"]}, index=f.index),
             f[["temp", "rain", "wind", "hum"]]],
            axis=1,
        )[meta["features_B"]]
        cf = np.clip(model.predict(X), 0, 1)
        cf[X["sun_alt"].values == 0] = 0
        preds.append(cf)
    cf = np.mean(preds, axis=0)
    return pd.DataFrame({"datetime": f["datetime"], "pred_kwh": cf * capacity_kw}), sites
