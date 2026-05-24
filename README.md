# 전력수요 실시간 예측 데모

서울 기상 데이터와 LightGBM 모델을 사용해 현재 전력수요를 예측하고, 한국전력거래소 현재 전력수급 API의 실제 수요와 비교하는 학습 프로젝트용 웹앱입니다.

## 데이터 출처

- 전력수급 데이터: 한국전력거래소 / 공공데이터포털
- 기상 데이터: Open-Meteo

본 예측 결과는 학습 프로젝트 목적의 참고 정보이며 실제 전력계통 운영 판단에 사용하면 안 됩니다.

## Streamlit 배포

1. 이 폴더의 파일을 GitHub 저장소에 업로드합니다.
2. Streamlit Community Cloud에서 이 저장소를 연결합니다.
3. Main file path는 `streamlit_app.py`로 설정합니다.
4. Streamlit Secrets에 아래 값을 추가합니다.

```toml
KPX_SERVICE_KEY = "여기에_본인_KPX_API_키"
```

## 로컬 실행

```powershell
pip install -r requirements_deploy.txt
$env:KPX_SERVICE_KEY="여기에_본인_KPX_API_키"
streamlit run streamlit_app.py
```

## 공개 기간

앱 코드는 `2026-06-24` 이후 자동 비활성화되도록 설정되어 있습니다.
