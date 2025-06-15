# data_preprocessing - AI Hub 졸음운전 통제환경 데이터 전처리

AI Hub에서 제공하는 졸음운전 예방 통제환경 데이터셋을 사용하여 새롭게 모델을 구성한다.  
- 원천 이미지와 JSON 라벨 데이터를 기반으로
- 눈 영역(`Leye`, `Reye`)을 Bounding Box로 시각화
- 좌우 눈 좌표가 정상적으로 매핑되어 있음을 확인


이를 기준으로
- JSON의 좌표를 이용한 좌우 눈 crop
- 흑백 + 리사이즈
- 라벨에 따라 `open/closed` 분류 및 저장 예정