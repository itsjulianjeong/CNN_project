# data 폴더
이 폴더는 졸음운전 감지를 위한 CNN 모델 학습에 사용된 데이터셋을 저장합니다.  
단, 전체 이미지는 48,000 장이므로 업로드 하지 않았습니다.  
향후 학습용으로 불러올 때는 `processed` 디렉토리를 기준으로 진행합니다.  

## 사용한 데이터셋  
- **출처**: [Kaggle - Drowsiness Detection Dataset](https://www.kaggle.com/datasets/kutaykutlu/drowsiness-detection?select=closed_eye)  
- **원본 데이터**: MRL Eye Dataset 일부 발췌본  
- **전체 MRL Eye Dataset**: [공식 사이트](http://mrl.cs.vsb.cz/eyedataset)  

## 데이터셋 설명  
- 눈 이미지를 분류하기 위한 용도로 준비된 MRL Eye Dataset의 일부입니다.  
- 적외선 카메라로 다양한 조명 조건과 센서 환경에서 촬영되었으며, low/high 해상도 이미지를 포함하고 있습니다.  
- 컬러가 아닌 **흑백(grayscale)** 이미지입니다.  
- 눈 부분만 crop되어 있는 상태로 제공됩니다. (즉, 얼굴 전체가 아닌 눈 주변만 존재)  

## 디렉토리 구조  
`raw`와 `processed`로 구성했으나, 이는 단순히 다양한 데이터셋들을 전처리하면서 디렉토리 구조를 유지하기 위함이며, 최종으로 설정한 데이터셋은 이미 crop되어 있는 이미지였기에 `raw`에서 `processed`로 move만 했습니다.  
- `notebooks/data_preprocessing/drowsiness_cls_final.ipynb` 파일 참고  
- `raw/`
  - `closed_eye`: 감은 눈 이미지 24,000장  
  - `open_eye`: 뜬 눈 이미지 24,000장
- `processed/`
  - `Closed`: 감은 눈 이미지 24,000장  
  - `Open`: 뜬 눈 이미지 24,000장  