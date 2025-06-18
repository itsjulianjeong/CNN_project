# 졸음운전 방지를 위한 실시간 눈 감김 감지 시스템 (CNN 기반)

CNN 기반의 눈 상태 분류 모델을 활용해 졸음운전 상태를 실시간으로 감지하는 시스템입니다.  
눈이 일정 시간 이상 감긴 경우 경고음을 울리며, 해당 화면을 캡처하여 저장합니다.  

---

# 데이터셋
- 출처: [Kaggle - Drowsiness Detection Dataset](https://www.kaggle.com/datasets/kutaykutlu/drowsiness-detection?select=closed_eye)
- 내용:
  - 눈을 감은(Closed) 이미지 24,000장  
  - 눈을 뜬(Open) 이미지 24,000장  
  - 흑백 이미지  
  - 눈 영역만 Crop된 상태  

---

# 프로젝트 구조
CNN_project/  
├── data/  
│ ├── raw/ # 원본 이미지 (Kaggle)  
│ └── processed/ # 학습용 전처리 데이터  
├── notebooks/  
│ ├── data_preprocessing/ # 전처리 노트북  
│ └── model_training/ # CNN 학습 노트북  
├── results/  
│ ├── images/ # 시각화 결과 (loss, confusion matrix 등)  
│ └── reports/ # classification_report.txt 저장  
├── webcam_app/  
│ ├── model/ # 학습된 모델 파일 (.keras)  
│ ├── captures/ # 졸음 상태 감지 시 스크린샷 저장 폴더  
│ ├── utils/ # 전처리, 시각화, 경고음 재생 유틸  
│ │ ├── image_utils.py  
│ │ ├── draw_result.py  
│ │ ├── beep.mp3  
│ │ └── \__init\__.py  
│ ├── \__init\__.py   
│ └── predictor.py # 실시간 예측 메인 코드  
├── drowsiness_detector.py # 실행 코드  
└── README.md  

---

# 실시간 졸음 감지 기능
## Detection 방식
- MediaPipe FaceMesh로 눈 랜드마크 추출 -> 좌우 눈 Crop  
- 훈련된 CNN 모델로 open/closed를 프레임마다 추정  
- 양쪽 눈이 일정 시간 이상 감긴 경우 경고 처리  

## 경고 조건
양쪽 눈이 0.7 미만 확률로 15프레임 이상 감긴 경우  
- 화면 캡처 저장 (webcam_app/captures/)  
- 경고음 발생 (beep.mp3)  
- 빨간색 반투명 오버레이 표시  