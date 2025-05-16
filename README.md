# OpenCV + ONNX 기반 불량 검출 실습 프로젝트

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue?logo=opencv)
![OpenCV DNN](https://img.shields.io/badge/OpenCV-DNN_Module-green?logo=opencv)
![ONNX](https://img.shields.io/badge/ONNX-ready-lightblue?logo=onnx)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c?logo=pytorch)
![NumPy](https://img.shields.io/badge/Library-NumPy-013243?logo=numpy)
![Jupyter](https://img.shields.io/badge/Tool-Jupyter-F37626?logo=jupyter)
![Visual Studio 2022](https://img.shields.io/badge/IDE-Visual%20Studio%202022-blueviolet?logo=visualstudio)
![ChatGPT](https://img.shields.io/badge/AI-ChatGPT-10a37f?logo=openai&logoColor=white)
![GitHub Copilot](https://img.shields.io/badge/AI-GitHub_Copilot-blue?logo=github&logoColor=white)

---
## 최종 소감
> 단순한 모델 학습을 넘어, Python과 C++ 양쪽에서 ONNX 모델을 활용한 실전 추론까지 직접 구현해보며 모델 전환과 활용 흐름을 처음부터 끝까지 경험할 수 있었다.
Jupyter에서 PyTorch로 기본 CNN과 ResNet18 모델을 실험하며 성능 차이를 체감했고, 이를 ONNX로 내보내어 OpenCV DNN 모듈로 불러오는 과정까지 성공적으로 연결했다.
또한, C++에서는 단순 추론을 넘어서 X-ray 스타일 필터, 외곽선 검출, 결함 시각화, 템플릿 매칭까지 시도해보며 OpenCV의 다양한 이미지 처리 기법을 실제 코드로 체득할 수 있었다.
5일이라는 짧은 기간이었지만, 학습-내보내기-불러오기-결함 판단이라는 일련의 흐름을 혼자서 완성해내며 모델링부터 시각화까지 딥러닝과 영상처리의 연결 고리를 스스로 만들 수 있게 된 값진 시간이었다.

## 프로젝트 소개 (2025.04.20 ~ 2025.04.24)

본 프로젝트는 ChatGPT를 참고하고 GitHub Copilot의 도움을 받아, OpenCV와 C++를 활용한 이미지 처리 및 불량 검출을 5일간 실습한 것입니다.

 1. OpenCV 기반의 전처리와 외곽선 검출을 통해 결함을 시각적으로 표현 
 
 2. 템플릿 매칭 기반 유사도 판별
 
 3. PyTorch로 학습한 CNN 분류 모델을 ONNX 형식으로 변환하여 C++ 환경에서 불량 검출

### 주요 기능
- X-ray 스타일 필터 (ColorMap 적용)
- 외곽선 기반 결함 시각화
- 템플릿 매칭 기반 유사도 판별
- ONNX 모델을 이용한 다중 클래스 분류 추론 (OpenCV DNN 모듈 사용)

---

## 결과 예시
### 1. OpenCV 기반의 전처리와 외곽선 검출 통해 결함을 시각적으로 표현 

<img src="./oepncvTest/refimg/result_good_001.png" alt="외곽선" width="200"/><img src="./oepncvTest/refimg/result_hole_003.png" alt="불량검출" width="200"/>

### 2. 템플릿 매칭 기반 유사도 판별

![기준 이미지](./oepncvTest/refimg/2025-04-24%20122121.jpg)

### 3. PyTorch로 학습한 CNN 분류 모델을 ONNX 형식으로 변환하여 C++ 환경에서 불량 검출

ONNX 모델은 숫자 인덱스로 클래스를 반환하므로, 아래와 같이 매핑됩니다:
```python
{'crack': 0, 'cut': 1, 'good': 2, 'hole': 3, 'print': 4}
```
![기준 이미지](./oepncvTest/refimg/2025-04-24%20122407.jpg)

## 딥러닝 기반 분류 모델 실험

처음에는 PyTorch로 간단한 CNN 모델을 구현하여 hazelnut 데이터셋을 분류하였습니다.  
아래는 에폭 수에 따른 정확도 변화입니다:

| 모델 | 에폭 | 테스트 정확도 |
|------|------|----------------|
| 기본 CNN (10 Epoch) | 10 | 약 25% |
| 기본 CNN (30 Epoch) | 30 | 약 5% |
| ✅ ResNet18 (사전학습) | 10 | 100% |

기본 CNN 모델은 구조가 단순한 데다, 데이터셋 크기 자체도 부족하여,  
오히려 학습이 진행될수록 정확도가 낮아지는 현상이 나타났습니다.

이에 따라 사전학습된 ResNet18 모델을 도입하고,  
클래스 수를 5개로 설정한 뒤 마지막 레이어만 수정하여 재학습을 진행한 결과,  
테스트셋에서 100% 정확도를 얻을 수 있었습니다.

## 폴더 구조

```text
opencvTest/
├── hazelnut/
│   ├── train/               # 테스트 이미지(단순하게 정확도 측정을 위해 반대로 하였음)
│   │   └── good/            
│   └── test/                # 학습용 이미지(분류해서 훈련시키기 위해 반대로 하였음)
│       ├── good/
│       ├── crack/
│       ├── cut/
│       ├── hole/
│       └── print/
├── opencvTest1.cpp          # C++ 메인 추론 코드
├── hazel_model.onnx         # PyTorch → ONNX 모델
├── results/                 # 추론 결과 이미지 저장
└── Hazelnut Cnn Train.ipynb # CNN 학습용 Jupyter Notebook
```

## 데이터셋 출처

본 프로젝트는 MVTec Anomaly Detection Dataset (MVTec AD)를 사용합니다.

- Dataset: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- License: CC BY-NC-SA 4.0



