# Streamlit 무료 배포 절차

이 앱은 학습 프로젝트용 실시간 전력수요 예측 데모입니다.

## 포함된 안전장치

- KPX API 키는 코드에 저장하지 않고 Streamlit Secrets로만 사용합니다.
- 실시간 API 호출 결과는 화면에만 표시하고 파일/DB에 저장하지 않습니다.
- 데이터 출처와 참고용 고지를 앱 화면에 표시합니다.
- `2026-06-24` 이후 앱이 자동 비활성화되도록 설정했습니다.

## 배포 파일

- `streamlit_app.py`
- `requirements_deploy.txt`
- `models/lightgbm_power_demand_none.pkl`
- `collect_current_power_supply.py`
- `realtime_power_app.py`
- `train_lightgbm_power_demand.py`

## Streamlit Community Cloud 설정

1. 이 프로젝트를 GitHub 저장소에 올립니다.
2. Streamlit Community Cloud에서 `New app`을 선택합니다.
3. Repository와 branch를 선택합니다.
4. Main file path에 아래 값을 넣습니다.

```text
streamlit_app.py
```

5. Advanced settings의 Secrets에 아래처럼 API 키를 넣습니다.

```toml
KPX_SERVICE_KEY = "여기에_본인_KPX_API_키"
```

6. Deploy를 누릅니다.

## 주의

- API 키를 GitHub에 직접 올리지 마세요.
- 과도한 자동 새로고침은 Open-Meteo와 KPX API 호출 제한에 걸릴 수 있습니다.
- 이 앱은 참고용 예측 데모이며 실제 전력계통 운영 판단에 사용하면 안 됩니다.
