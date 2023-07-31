![타이틀](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/c9dbff59-f53a-4222-a0fd-9f4d772e5cb3)

## **A. 목차**
  * A. [목차](#a-목차)
  * B. [프로젝트 진행](#b-프로젝트-진행)
    * B.1. [문제 정의](#b1-문제-정의)
    * B.2. [주요 액션](#b2-주요-액션)
      * B.2.1. [검수 절차 정의](#b21-검수-절차-정의)
      * B.2.2. [학습 및 테스트용 동영상 촬영](#b22-학습-및-테스트용-동영상-촬영)
      * B.2.3. [어노테이션 (Bounding Box) 및 영상 분할](#b23-어노테이션-bounding-box-및-영상-분할)
      * B.2.4. [객체 인식을 위한 YOLOv5s 모델 개발](#b24-객체-인식을-위한-yolov5s-모델-개발) - [관련 코드](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/blob/970e9b330606b5a45c64293e6b4578875c08f6a2/YOLOv5s_train_test/YOLOv5s_train_test.ipynb) | [모델 테스트 결과](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/tree/970e9b330606b5a45c64293e6b4578875c08f6a2/YOLOv5s_train_test/test_results)
      * B.2.5. [행동 인식을 위한 LSTM 모델 개발](#b25-행동-인식을-위한-lstm-모델-개발) - [관련 코드](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/blob/03c117013ea6285c95d51954e85bbcbbabbec976/LSTM_train_test.ipynb)
      * B.2.6. [검수 절차 구축 및 PC APP 개발](#b26-검수-절차-구축-및-pc-app-개발) -[PC 어플리케이션 다운로드](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/blob/62319cd54da9bda4cad1abef90abed321749d7e0/Inspection-Model_Application.zip) | [관련 코드](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/blob/a935d7a07596d7449443c474fa150c06c6972291/inspection_process.py)
  * C. 결과 및 기대효과
  * D. Deck
  * E. Methods Used
--- 
## **B. 프로젝트 진행**

### **B.1. 문제 정의**
* 매년 개인 간 거래(C2C) 시장 규모가 확대되고 있으나, '사기 거래'에 따른 거래 분쟁 또한 심화됨
* 고가 제품을 다루는 리셀 플랫폼을 중심으로 검수 시스템이 운영되고 있으나, 전문 인력에 대한 의존성이 높고 일부 플랫폼 및 한정된 품목에서만 검수가 이루어지는 한계가 있음
* 본 프로젝트에서는 **판매자가 물품의 모든 면을 촬영하고 딥러닝 모델이 제품의 상태와 손동작을 확인하는 검수 절차**를 구축하여 플랫폼의 업무 과중을 줄이고 거래의 신뢰도를 높이고자 함

### **B.2. 주요 액션**
* 본 프로젝트에서는 검수 물품으로 아래 사진과 같이 뒤집으면 표정이 바뀌는 문어 인형을 사용하였음<br>
<p align = "center"><img src = https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/beaba5fa-203d-4b15-acf8-aff297f70546 width = 70% height = 70%></p>

#### **B.2.1. 검수 절차 정의**
* 본 프로젝트에서 구현하고자 하는 검수 절차를 아래와 같이 크게 5단계로 정의함
* 각 단계에서 요구하는 물품의 상태와 손동작이 모두 인식되어야 다음 단계로 넘어가고, 가장 마지막인 5단계에 검수 결과를 안내받음
* 윗면과 아랫면을 보여주는 손동작에서는 Mediapipe를 통한 랜드마크 검출에 어려움이 있어 LSTM 모델을 사용하지 않음
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

* YOLO 학습 및 테스트 코드는 [링크_1](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/blob/970e9b330606b5a45c64293e6b4578875c08f6a2/YOLOv5s_train_test/YOLOv5s_train_test.ipynb)을, 테스트 결과(Confusion Matrix 등)는 [링크_2](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/tree/970e9b330606b5a45c64293e6b4578875c08f6a2/YOLOv5s_train_test/test_results)를 참고해주시기 바랍니다.
* Augmentation 적용 후 확보한 총 14,494장의 바운딩 박스 데이터셋을 학습하여 불량을 포함한 총 7개의 객체를 검출할 수 있도록 함
* 뒤집으면 반대의 표정이 나오는 인형의 특징을 활용하여 찡그린 표정을 '불량'이라고 정의하였으며, <br>불량을 제외한 나머지 6개의 객체는 다음과 같음
  
* 모델 성능을 테스트한 결과 Precision-Recall Curve와 Confusion Matrix는 다음과 같음

<table align ='center'>
 <tr>
  <th>Precision-Recall Curve</th>
  <th>Confusion Matrix</th>
 </tr>
 <tr>
  <td><p align = 'center'><img src = "https://github.com/TAEJIN-AHN/AI-Doll-Inspection/blob/main/YOLOv5s_train_test/test_results/PR_curve.png"></p></td>
  <td><p align = 'center'><img src = "https://github.com/TAEJIN-AHN/AI-Doll-Inspection/blob/main/YOLOv5s_train_test/test_results/confusion_matrix.png" width  = 90% height = 90%></p></td>
 </tr>
</table>

* 빠른 연산을 위해 모델을 onnx 형식으로 변환하여 추론에 사용

#### **B.2.5. 행동 인식을 위한 LSTM 모델 개발**

<p align = 'center'><img src = 'https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/c812af74-cbd0-444f-b47e-d1f5cd6c7915' width = 80% height = 80%></p>
<p align = 'center'>※ 단, 손의 랜드마크는 YOLO가 검출한 손의 Bounding Box 안에서만 확인할 수 있도록 함</p>

* 자세한 코드와 내용은 [링크](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/blob/03c117013ea6285c95d51954e85bbcbbabbec976/LSTM_train_test.ipynb)를 참고해주시기 바랍니다.
* 개발한 LSTM 모델의 성능은 다음과 같음

 <table>
   <tr>
    <th>내려놓기(하강/멈춤)</th>
    <th>회전하기(회전/기타)</th>
   </tr>
  <tr>
   <td><p align = 'center' valign = 'center'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/beae91f1-529d-4e24-a431-81f287fe3785" width = 95% height = 95%>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</p></td>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/6d52bbd7-4186-430f-937c-7063a7716944" width = % height = 100%></p></td>     
  </tr>
  <tr>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/6136e27e-9994-478d-9138-31f56870f088" width = 100% height = 100%></p></td>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/2ffeea7c-cd13-4a05-bd91-44e152f5214c" width = 100% height = 100%></p></td>     
  </tr> 
</table>

#### **B.2.6. 검수 절차 구축 및 PC APP 개발**

<p align ='center'><a href="https://youtu.be/T3MaxBySd3U/" target="_blank"><img alt="youtube_link" width = 60% height = 60% src="http://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/0507e2dd-32ef-46fc-afd7-810b37a8d506"></a></p>

* PC APP 전체 파일은 [링크_1](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/blob/62319cd54da9bda4cad1abef90abed321749d7e0/Inspection-Model_Application.zip)를 통해 내려받을 수 있고, 그 중 검수 절차 코드는 [링크_2](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/blob/a935d7a07596d7449443c474fa150c06c6972291/inspection_process.py)를 통해 확인 가능합니다.
* 앞선 단계에서 학습한 YOLOv5s, LSTM 모델을 이식하여 행동 및 절차 인식 알고리즘을 구축함
* 또한, 해당 알고리즘을 편리하게 사용할 수 있도록 PyQt를 사용하여 PC 어플리케이션을 개발함
* 해당 알고리즘이 진행되기 위한 주요한 전제 조건과 행동 판단 조건, 절차 인식 조건은 다음과 같음

<table align = 'center'>
 <tr>
  <td rowspan = '2' align = 'center' valign = 'center'>전제 조건</td>
  <td>반드시 한 손으로만 행동을 수행한다</td>
 </tr>
 <tr>
  <!--<td>전제 조건</td>-->
  <td>각 행동(물건 내려놓기, 물건 회전하기)의 수행시간은 1초 내외이다</td>
 </tr>
 <tr>
  <td rowspan = '3' align = 'center' valign = 'center'>행동 판단 조건</td>
  <td>중앙의 적색 사각형 안에 손이 처음 감지될 때부터 사각형 밖으로 손이 나갈 때까지의 전체 프레임 수를 확인한다</td>
 </tr>
 <tr>
  <!--<td>행동 판단 조건 </td>-->
  <td>각 단계에서 필요로 하는 동작을 탐지※한 프레임의 수가 전체 중 일정 비율 이상일 시 행동을 수행하였다고 판단한다</td>
 </tr>
 <tr>
  <!--<td>행동 판단 조건 </td>-->
  <td>행동이 수행되지 않았다고 판단될 때에는 사용자에게 재수행을 요구한다</td>
 </tr>
 <tr>
  <td rowspan = '3' align = 'center' valign = 'center'>절차 인식 조건</td>
  <td>정면(내려놓기) → 옆면 및 뒷면(회전하기) → 윗면 → 아랫면 순으로 검수를 진행한다</td>
 </tr>
 <tr>
  <!--<td>절차 인식 조건 </td>-->
  <td>물건의 상태나 손동작이 각 단계에서 요구하는 조건을 충족하지 못할 경우 재수행을 요구한다</td>
 </tr>
 <tr>
  <!--<td>절차 인식 조건 </td>-->
  <td>아랫면까지 모두 확인한 후에는 불량 여부를 안내한다</td>
 </tr>
</table>

<p align = 'center'>※<a href = 'https://github.com/TAEJIN-AHN/AI-Doll-Inspection/blob/87330a3ab0d3509e41147cef7a7cfbec7046f950/proba_test.md'>링크</a>와 같이 훈련 영상과 테스트 영상을 참고하여 행동 탐지의 기준이 되는 Probability Threshold를 정함</p>
