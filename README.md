# 수중 위험 감지 시스템 (Marine Danger Alert System)

## 프로젝트 개요
드론 영상을 활용하여 수중의 위험 요소(상어, 독성 해파리 등)를 실시간으로 탐지하고, 객체의 접근 속도(성장률)를 분석하여 위험도를 판별하는 시스템입니다. OpenVINO를 활용하여 추론 속도를 최적화하고, Streamlit 기반의 웹 대시보드로 시각화합니다.

## 디렉토리 구조
```text
Marine_Safety_Project/
├── app.py                      # Streamlit 웹 애플리케이션 실행 파일
├── best_openvino_model/        # OpenVINO 변환 모델 폴더 (.xml, .bin)
├── data/                       # 테스트용 샘플 동영상 폴더
├── requirements.txt            # 필수 라이브러리 목록
└── README.md                   # 프로젝트 설명 문서

요구 사항 및 설치 방법
본 프로젝트는 Python 3.12 환경에서 테스트되었습니다.

가상환경 생성 및 활성화 (권장)

Bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
의존성 설치

Bash
pip install -r requirements.txt
실행 방법 (코드 재현성)
app.py와 best_openvino_model 폴더가 동일한 디렉토리에 위치해야 합니다.

다음 명령어를 실행하여 Streamlit 웹 애플리케이션을 구동합니다.

streamlit run app.py

브라우저에서 http://localhost:8501로 접속하여 동영상 파일(mp4, avi 등)을 업로드하면 실시간 분석이 시작됩니다.

주요 기능
고속 객체 탐지: YOLO11 모델을 OpenVINO 포맷으로 경량화하여 실시간 분석.

동적 위험도 산출: 감지된 객체의 크기 변화(성장률)를 계산하여 접근 속도 기반의 위험도(안전/주의/경고/위험) 분류.

실시간 관제 대시보드: 유튜브 스타일의 재생 제어(5초 건너뛰기, 일시정지, 타임라인 슬라이더)와 상태 요약 정보 제공.

평가 항목 관련 정보
데이터셋: (사용하신 데이터셋 간략히 설명, 예: Roboflow를 활용하여 약 N장의 라벨링 일관성 확보)

성능: mAP@0.5 OO% 달성

모델 크기: 원본 (OO MB) -> OpenVINO (OO MB)로 최적화

위 내용들을 바탕으로, 실제 프로젝트에서 얻은 결과(mAP 수치, 모델 용량 등)를 채워 넣어 보고서의 완성도를 높이시기 바랍니다.