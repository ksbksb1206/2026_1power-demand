import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from collect_current_power_supply import fetch_xml, parse_xml
from train_lightgbm_power_demand import add_cyclical_time_features


HOST = "127.0.0.1"
PORT = 8000
MODEL_PATH = "models/lightgbm_power_demand_none.pkl"
CURRENT_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
SEOUL_LATITUDE = 37.5665
SEOUL_LONGITUDE = 126.9780

WEATHER_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
    "pressure_msl",
    "surface_pressure",
]


HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>전력수요 실시간 예측</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --line: #d8dee9;
      --text: #172033;
      --muted: #647084;
      --accent: #0b6bcb;
      --danger: #b42318;
      --ok: #067647;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, "Malgun Gothic", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 24px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { margin: 0; font-size: 24px; letter-spacing: 0; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 16px;
    }
    button {
      border: 1px solid #0959aa;
      background: var(--accent);
      color: white;
      border-radius: 6px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled { opacity: .65; cursor: wait; }
    .status { color: var(--muted); font-size: 14px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .metric, .section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .label { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
    .value { font-size: 28px; font-weight: 800; white-space: nowrap; }
    .unit { font-size: 14px; color: var(--muted); margin-left: 4px; }
    .good { color: var(--ok); }
    .bad { color: var(--danger); }
    .section { margin-top: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { text-align: left; padding: 9px 8px; border-bottom: 1px solid var(--line); }
    th { color: var(--muted); font-weight: 700; }
    .two { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    @media (max-width: 860px) {
      main { padding: 16px; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .two { grid-template-columns: 1fr; }
      .value { font-size: 23px; }
      .toolbar { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <h1>전력수요 실시간 예측</h1>
  </header>
  <main>
    <div class="toolbar">
      <button id="refresh">새로고침</button>
      <div class="status" id="status">대기 중</div>
    </div>

    <div class="grid">
      <div class="metric">
        <div class="label">예측 전력수요</div>
        <div class="value"><span id="predicted">-</span><span class="unit">MW</span></div>
      </div>
      <div class="metric">
        <div class="label">현재 전력수요</div>
        <div class="value"><span id="actual">-</span><span class="unit">MW</span></div>
      </div>
      <div class="metric">
        <div class="label">절대 오차</div>
        <div class="value"><span id="absError">-</span><span class="unit">MW</span></div>
      </div>
      <div class="metric">
        <div class="label">절대 오차율</div>
        <div class="value"><span id="ape">-</span><span class="unit">%</span></div>
      </div>
    </div>

    <div class="two">
      <section class="section">
        <table>
          <thead><tr><th>기상 변수</th><th>값</th></tr></thead>
          <tbody id="weather"></tbody>
        </table>
      </section>
      <section class="section">
        <table>
          <thead><tr><th>정보</th><th>값</th></tr></thead>
          <tbody id="meta"></tbody>
        </table>
      </section>
    </div>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const refresh = document.getElementById("refresh");

    function fmt(value, digits = 1) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toLocaleString("ko-KR", {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits
      });
    }

    function rows(obj, units = {}) {
      return Object.entries(obj).map(([key, value]) => {
        const unit = units[key] ? " " + units[key] : "";
        return `<tr><td>${key}</td><td>${fmt(value, 2)}${unit}</td></tr>`;
      }).join("");
    }

    async function loadPrediction() {
      refresh.disabled = true;
      statusEl.textContent = "API 호출 및 예측 중...";
      try {
        const res = await fetch("/api/predict", { cache: "no-store" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "prediction failed");

        document.getElementById("predicted").textContent = fmt(data.prediction_mw, 1);
        document.getElementById("actual").textContent = fmt(data.actual_current_demand_mw, 1);
        document.getElementById("absError").textContent = fmt(data.absolute_error_mw, 1);
        document.getElementById("ape").textContent = fmt(data.absolute_percentage_error, 2);
        document.getElementById("ape").className = data.absolute_percentage_error <= 5 ? "good" : "bad";

        document.getElementById("weather").innerHTML = rows(data.weather, {
          temperature_2m: "C",
          relative_humidity_2m: "%",
          apparent_temperature: "C",
          precipitation: "mm",
          wind_speed_10m: "km/h",
          shortwave_radiation: "W/m2",
          cloud_cover: "%",
          pressure_msl: "hPa",
          surface_pressure: "hPa",
          discomfort_index: "",
          cooling_degree: "",
          heating_degree: ""
        });
        document.getElementById("meta").innerHTML = `
          <tr><td>기상 기준시각</td><td>${data.weather_time}</td></tr>
          <tr><td>전력 기준시각</td><td>${data.power_base_datetime}</td></tr>
          <tr><td>모델</td><td>${data.model_path}</td></tr>
          <tr><td>비교 방식</td><td>${data.comparison_note}</td></tr>
        `;
        statusEl.textContent = "업데이트 완료: " + new Date().toLocaleString("ko-KR");
      } catch (err) {
        statusEl.textContent = "오류: " + err.message;
      } finally {
        refresh.disabled = false;
      }
    }

    refresh.addEventListener("click", loadPrediction);
    loadPrediction();
  </script>
</body>
</html>
"""


MODEL_BUNDLE = None


def load_model_bundle():
    global MODEL_BUNDLE
    if MODEL_BUNDLE is None:
        MODEL_BUNDLE = joblib.load(MODEL_PATH)
    return MODEL_BUNDLE


def fetch_json(url, params):
    query = urllib.parse.urlencode(params, doseq=True)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_current_seoul_weather():
    params = {
        "latitude": SEOUL_LATITUDE,
        "longitude": SEOUL_LONGITUDE,
        "current": WEATHER_FIELDS,
        "hourly": "temperature_2m",
        "forecast_days": 1,
        "timezone": "Asia/Seoul",
    }
    data = fetch_json(CURRENT_WEATHER_URL, params)
    current = data["current"]
    row = {field: current.get(field) for field in WEATHER_FIELDS}
    row["datetime"] = pd.to_datetime(current["time"])

    hourly = pd.DataFrame(data["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    same_day = hourly[hourly["time"].dt.date == row["datetime"].date()]
    row["daily_max_temperature"] = same_day["temperature_2m"].max()
    row["daily_min_temperature"] = same_day["temperature_2m"].min()
    row["daily_mean_temperature"] = same_day["temperature_2m"].mean()
    return row, current["time"]


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
    df = add_cyclical_time_features(df)
    return df


def predict_realtime():
    service_key = os.getenv("KPX_SERVICE_KEY")
    if not service_key:
        raise RuntimeError("KPX_SERVICE_KEY environment variable is missing.")

    bundle = load_model_bundle()
    model = bundle["model"]
    features = bundle["features"]

    weather_row, weather_time = fetch_current_seoul_weather()
    feature_frame = add_realtime_features(weather_row)

    missing = [feature for feature in features if feature not in feature_frame.columns]
    if missing:
        raise RuntimeError(f"Missing model features: {missing}")

    prediction = float(model.predict(feature_frame[features])[0])

    power_row = parse_xml(fetch_xml(service_key))
    actual = float(power_row["currPwrTot"])
    error = actual - prediction
    abs_error = abs(error)
    ape = abs_error / abs(actual) * 100 if actual else None

    weather_payload = {
        field: float(feature_frame.iloc[0][field])
        for field in WEATHER_FIELDS
        if field in feature_frame.columns
    }
    for field in ["discomfort_index", "cooling_degree", "heating_degree"]:
        weather_payload[field] = float(feature_frame.iloc[0][field])

    return {
        "prediction_mw": prediction,
        "actual_current_demand_mw": actual,
        "error_mw": error,
        "absolute_error_mw": abs_error,
        "absolute_percentage_error": ape,
        "weather": weather_payload,
        "weather_time": weather_time,
        "power_base_datetime": str(power_row["baseDatetime"]),
        "model_path": MODEL_PATH,
        "comparison_note": "Weather-only hourly model prediction compared with latest KPX current demand.",
    }


class RealtimePowerHandler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send(200, HTML, "text/html; charset=utf-8")
            return
        if self.path == "/api/predict":
            try:
                payload = predict_realtime()
                self._send(
                    200,
                    json.dumps(payload, ensure_ascii=False),
                    "application/json; charset=utf-8",
                )
            except Exception as exc:
                self._send(
                    500,
                    json.dumps({"error": str(exc)}, ensure_ascii=False),
                    "application/json; charset=utf-8",
                )
            return
        self._send(404, "Not found", "text/plain; charset=utf-8")

    def log_message(self, format, *args):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {format % args}")


def main():
    if not Path(MODEL_PATH).exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}")

    server = ThreadingHTTPServer((HOST, PORT), RealtimePowerHandler)
    print(f"Realtime power app running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
