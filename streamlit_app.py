from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from collect_current_power_supply import fetch_xml, parse_xml
from realtime_power_app import fetch_current_seoul_weather, MODEL_PATH, WEATHER_FIELDS
from train_lightgbm_power_demand import add_cyclical_time_features


APP_EXPIRES_ON = date(2026, 6, 24)


def get_kpx_service_key():
    if "KPX_SERVICE_KEY" in st.secrets:
        return st.secrets["KPX_SERVICE_KEY"]
    return None


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


def add_realtime_features(weather_row):
    df = pd.DataFrame([weather_row])
    df["datetime"] = pd.to_datetime(df["datetime"])
    temp = df["temperature_2m"]
    humidity = df["relative_humidity_2m"]
    df["discomfort_index"] = (
        0.81 * temp
        + 0.01 * humidity * (0.99 * temp - 14.3)
        + 46.3
    )
    df["cooling_degree"] = (temp - 24).clip(lower=0)
    df["heating_degree"] = (18 - temp).clip(lower=0)
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["season"] = ((df["month"] % 12) // 3 + 1)
    return add_cyclical_time_features(df)


def predict_once(service_key):
    bundle = load_model_bundle()
    model = bundle["model"]
    features = bundle["features"]

    weather_row, weather_time = fetch_current_seoul_weather()
    feature_frame = add_realtime_features(weather_row)
    prediction = float(model.predict(feature_frame[features])[0])

    power_row = parse_xml(fetch_xml(service_key))
    actual = float(power_row["currPwrTot"])
    absolute_error = abs(actual - prediction)
    absolute_percentage_error = absolute_error / abs(actual) * 100

    weather_values = {
        field: float(feature_frame.iloc[0][field])
        for field in WEATHER_FIELDS
        if field in feature_frame.columns
    }
    for field in ["discomfort_index", "cooling_degree", "heating_degree"]:
        weather_values[field] = float(feature_frame.iloc[0][field])

    return {
        "prediction": prediction,
        "actual": actual,
        "absolute_error": absolute_error,
        "absolute_percentage_error": absolute_percentage_error,
        "weather_time": weather_time,
        "power_time": str(power_row["baseDatetime"]),
        "weather": weather_values,
    }


st.set_page_config(
    page_title="전력수요 실시간 예측",
    page_icon="",
    layout="wide",
)

st.title("전력수요 실시간 예측")

today = date.today()
if today > APP_EXPIRES_ON:
    st.error("이 학습용 데모 앱은 공개 기간이 종료되었습니다.")
    st.caption("전력수급 데이터: 한국전력거래소/공공데이터포털 · 기상 데이터: Open-Meteo")
    st.stop()

st.caption(
    "전력수급 데이터: 한국전력거래소/공공데이터포털 · "
    "기상 데이터: Open-Meteo · 예측 결과는 학습 프로젝트 목적의 참고 정보입니다."
)
st.info(
    f"이 데모는 {APP_EXPIRES_ON.isoformat()} 이후 자동 비활성화됩니다. "
    "실시간 호출 결과는 화면에만 표시하며 저장하지 않습니다."
)

service_key = get_kpx_service_key()
if not service_key:
    st.error("KPX_SERVICE_KEY가 Streamlit Secrets에 설정되어 있지 않습니다.")
    st.code('KPX_SERVICE_KEY = "your_api_key"', language="toml")
    st.stop()

if st.button("현재 데이터로 예측", type="primary") or "last_result" not in st.session_state:
    with st.spinner("현재 기상/전력수급 API 호출 및 예측 중..."):
        st.session_state.last_result = predict_once(service_key)
        st.session_state.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

result = st.session_state.last_result

col1, col2, col3, col4 = st.columns(4)
col1.metric("예측 전력수요", f"{result['prediction']:,.1f} MW")
col2.metric("현재 전력수요", f"{result['actual']:,.1f} MW")
col3.metric("절대 오차", f"{result['absolute_error']:,.1f} MW")
col4.metric("절대 오차율", f"{result['absolute_percentage_error']:.2f}%")

st.write(f"업데이트: {st.session_state.updated_at}")
st.write(f"기상 기준시각: {result['weather_time']} · 전력 기준시각: {result['power_time']}")

weather_df = pd.DataFrame(
    [{"variable": key, "value": value} for key, value in result["weather"].items()]
)
st.subheader("현재 서울 기상 및 파생변수")
st.dataframe(weather_df, use_container_width=True, hide_index=True)
