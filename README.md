![타이틀](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/c9dbff59-f53a-4222-a0fd-9f4d772e5cb3)

## **A. 목차**
  * A. [목차](#a-목차)
  * B. [프로젝트 진행](#b-프로젝트-진행)
    * B.1. [문제 정의](#b1-문제-정의)
    * B.2. [주요 액션](#b2-주요-액션)
      * B.2.1. [검수 절차 정의](#b21-검수-절차-정의)
      * B.2.2. [학습 및 테스트용 동영상 촬영](#b22-학습-및-테스트용-동영상-촬영)
      * B.2.3. [어노테이션 (Bounding Box) 및 영상 분할](#b23-어노테이션-bounding-box-및-영상-분할)
      * B.2.4. [객체 인식을 위한 YOLOv5s 모델 개발](#b24-객체-인식을-위한-yolov5s-모델-개발) - [관련 코드](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/blob/5727c88020701279ef4e4d02bb14553ea21704ad/yolov5s_train_test/YOLOv5s_train_test.ipynb) | [모델 테스트 결과](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/tree/5727c88020701279ef4e4d02bb14553ea21704ad/yolov5s_train_test/test_results)
      * B.2.5. 행동 인식을 위한 LSTM 모델 개발
      * B.2.6. YOLO와 LSTM 모델을 이식한 검수 절차 구축
      * B.2.7. PyQt를 활용한 검수 PC 어플리케이션 개발
  * C. 결과 및 기대효과
  * D. Deck
  * E. Methods Used
--- 
## **B. 프로젝트 진행**

### **B.1. 문제 정의**
* 매년 개인 간 거래(C2C) 시장 규모가 확대되고 있으나, '사기 거래'에 따른 거래 분쟁 또한 심화됨
* 고가 제품을 다루는 리셀 플랫폼을 중심으로 검수 시스템이 운영되고 있으나, 전문 인력에 대한 의존성이 높고 일부 플랫폼 및 한정된 품목에서만 검수가 이루어지는 한계가 있음
* 본 프로젝트에서는 판매자가 정해진 절차에 따라 물품을 촬영하고 딥러닝 모델이 제품의 상태와 손동작을 확인하는 검수 절차를 구축하여 플랫폼의 업무 과중을 줄이고 거래의 신뢰도를 높이고자 함

### **B.2. 주요 액션**
* 본 프로젝트에서는 검수 물품으로 아래 사진과 같은 문어 인형을 사용하였음<br>
<p align = "center"><img src = https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/beaba5fa-203d-4b15-acf8-aff297f70546 width = 70% height = 70%></p>

#### **B.2.1. 검수 절차 정의**
* 본 프로젝트에서 구현하고자 하는 검수 절차를 아래와 같이 크게 5단계로 정의함
* 윗면과 아랫면을 보여주는 손동작에서는 손바닥이 보이지 않기 때문에 랜드마크를 검출할 수 없어 LSTM 모델을 사용하지 않음
<p align = "center"><img src = https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/49e55063-80c7-4dac-8ff2-ef78155a6873 width = 50% height = 50%></p>

#### **B.2.2. 학습 및 테스트용 동영상 촬영**
* 검수 시나리오상 필요한 동작을 크게 4개로 구분지어 동영상 촬영
<table>
   <tr>
    <td><p align = 'center'>내려놓기</p></td>
    <td><p align = 'center'>회전하기</p></td>
    <td><p align = 'center'>윗면 보여주기</p></td>
    <td><p align = 'center'>아랫면 보여주기</p></td>
   </tr>
  <tr>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/cbfb5e18-b849-4dd0-bd90-fd76c5896e0c" alt="4" width = 80% height = 80%></p></td>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/53d203d0-65e3-46a4-a28a-675dd4102d85" alt="2" width = 80% height = 80%></p></td>     
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/b72733b2-c3df-4e42-80aa-e59efd854d37" alt="1" width = 80% height = 80%></p></td>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/a746869f-0ce0-49cf-b6ca-27c5d3085aa9" alt="3" width = 80% height = 80%></p></td>   
  </tr> 
</table>

#### **B.2.3. 어노테이션 (Bounding Box) 및 영상 분할**
* 촬영된 영상을 Adobe Premiere Pro를 활용하여 동작 단위(약 20프레임 길이)로 분할 → LSTM 모델 학습에 사용
* [Roboflow](https://roboflow.com/)의 Annotation Tool을 활용하여 프레임별 바운딩 박스 데이터셋을 확보 → YOLO 모델 학습에 사용
<p align = 'center'><img src = 'https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/d142fd9f-ebea-4448-a10c-02a371af4d28' width = 60% height = 60%></p>

#### **B.2.4. 객체 인식을 위한 YOLOv5s 모델 개발**
<p align = 'center'><img src = 'https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/35742e3a-9d50-4042-a63c-040edd3ddefe' width = 60% height = 60%></p>

* yolo 학습 및 테스트 코드는 [링크_1](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/blob/5727c88020701279ef4e4d02bb14553ea21704ad/yolov5s_train_test/YOLOv5s_train_test.ipynb)을, yolo 테스트 결과(Confusion Matrix 등)는 [링크_2](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/tree/5727c88020701279ef4e4d02bb14553ea21704ad/yolov5s_train_test/test_results)를 참고해주시기 바랍니다.
* Augmentation 적용 후 확보한 총 14,494장의 바운딩 박스 데이터셋을 학습하여 불량을 포함한 총 7개의 객체를 검출할 수 있도록 함
* 뒤집으면 반대의 표정이 나오는 인형의 특징을 활용하여 찡그린 표정을 '불량'이라고 정의하였으며, <br>불량을 제외한 나머지 6개의 객체는 다음과 같음
  
* 모델 성능을 테스트한 결과 Precision-Recall Curve와 Confusion Matrix는 다음과 같음

<table align ='center'>
 <tr>
  <th>Precision-Recall Curve</th>
  <th>Confusion Matrix</th>
 </tr>
 <tr>
  <td><p align = 'center'><img src = "https://github.com/TAEJIN-AHN/AI-Doll-Inspection/blob/main/yolov5s_train_test/test_results/PR_curve.png"></p></td>
  <td><p align = 'center'><img src = "https://github.com/TAEJIN-AHN/AI-Doll-Inspection/blob/main/yolov5s_train_test/test_results/confusion_matrix.png" width  = 90% height = 90%></p></td>
 </tr>
</table>

* 빠른 연산을 위해 모델을 onnx 형식으로 변환하여 추론에 사용

#### **B.2.5. 행동 인식을 위한 LSTM 모델 개발**
* 분할된 영상에 Mediapipe를 적용하여 손의 랜드마크를 검출하고 마디간 각도값 및 마디별 좌표값을 계산
* 이 때, 손의 랜드마크는 YOLO가 검출한 손의 Bounding Box 안에서만 확인할 수 있도록 함
* 본 프로젝트에서는 총 2개의 LSTM 모델을 개발하였으며 각각 학습한 데이터가 다름
  1. [내려놓기] 동작 시 손의 하강과 상승을 구분할 수 있는 모델 ← 마디별 **좌표값** 데이터
  2. [회전하기] 동작 시 손의 회전 여부를 확인할 수 있는 모델 ← 마디간 **각도값** 데이터
* 개발한 LSTM 모델의 성능은 다음과 같음
 <table>
   <tr>
    <td><p align = 'center'>내려놓기(상승/하강)</p></td>
    <td><p align = 'center'>회전하기(회전/기타)</p></td>
   </tr>
  <tr>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/6d52bbd7-4186-430f-937c-7063a7716944" alt="1" width = 80% height = 80%></p></td>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/6d52bbd7-4186-430f-937c-7063a7716944" alt="2" width = 80% height = 80%></p></td>     
  </tr>
  <tr>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/ad7af895-80ba-4058-acbd-4d8833811d02" alt="3" width = 80% height = 80%></p></td>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/ad7af895-80ba-4058-acbd-4d8833811d02" alt="4" width = 80% height = 80%></p></td>     
  </tr> 
</table>

#### **B.2.6. YOLO와 LSTM 모델을 이식한 검수 절차 구축**
* 
#### **B.2.7. PyQt를 활용한 검수 PC 어플리케이션 개발**
