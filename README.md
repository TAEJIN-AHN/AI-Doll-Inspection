![타이틀](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/c9dbff59-f53a-4222-a0fd-9f4d772e5cb3)

## **A. 목차**
  * A. 목차
  * B. 프로젝트 진행
    * B.1. 문제정의
    * B.2. 프로젝트 진행(주요 액션)
      * B.2.1. 학습 및 테스트용 동영상 촬영
      * B.2.2. 어노테이션 (Bounding Box) 및 영상 분할
      * B.2.3. 객체 인식을 위한 YOLOv5s 모델 개발
      * B.2.4. 행동 인식을 위한 LSTM 모델 개발
      * B.2.5. YOLO와 LSTM 모델을 이식한 검수 절차 구축
      * B.2.6. PyQt를 활용한 검수 PC 어플리케이션 개발
    * B.3. 결과
  * C. Deck
  * D. Methods Used
  * E. Contributing Members
--- 
## **B. 프로젝트 진행**

### **B.1. 문제정의**
* 매년 개인 간 거래(C2C) 시장 규모가 확대되고 있으나, '사기 거래'에 따른 거래 분쟁 또한 심화됨
* 고가 제품을 다루는 리셀 플랫폼을 중심으로 검수 시스템이 운영되고 있으나, 전문 인력에 대한 의존성이 높고 일부 플랫폼 및 한정된 품목에서만 검수가 이루어지는 한계가 있음
* 본 프로젝트에서는 판매자가 정해진 절차에 따라 물품을 촬영하고 딥러닝 모델이 제품의 상태와 손동작을 확인하는 검수 절차를 구축하여 플랫폼의 업무 과중을 줄이고 거래의 신뢰도를 높이고자 함

### **B.2. 주요액션**
* 본 프로젝트에서는 검수 물품으로 아래 사진과 같은 문어 인형을 사용하였음<br>
<p align = "center"><img src = https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/beaba5fa-203d-4b15-acf8-aff297f70546 width = 70% height = 70%></p>

* 뒤집으면 반대의 표정이 나오는 인형의 특징을 활용하여 찡그린 표정을 '불량'이라고 정의함

#### **B.2.1. 검수 절차 정의**
* 본 프로젝트에서 구현하고자 하는 검수 절차(시나리오)를 아래와 같이 정함 (※ 윗면과 아랫면을 보여주는 손동작에서는 손바닥이 보이지 않아 손의 관절별 랜드마크를 검출할 수 없어 인식 대상에서 제외)
<p align = "center"><img src = https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/6a064211-cd9f-4544-a99e-be454fcc6468 width = 50% height = 50%></p>

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
* 학습 및 테스트를 위해 촬영된 동영상을 Adobe Premiere Pro를 활용하여 동작 단위 (약 30프레임) 단위로 분할 → LSTM 모델 학습에 사용
* 컴퓨터 비전을 위한 오픈소스 플랫폼인 Roboflow의 Annotation Tool을 활용하여 프레임별 바운딩 박스 데이터셋을 확보 → YOLO 모델 학습에 사용
<p align = 'center'><img src = 'https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/d142fd9f-ebea-4448-a10c-02a371af4d28' width = 60% height = 60%></p>

#### **B.2.4. 객체 인식을 위한 YOLOv5s 모델 개발**
* 충분한 FPS (Frame Per Second) 확보를 위해 YOLOv5 모델 중 속도가 빠른 YOLOv5s 모델을 선택함  
* Augmentation 적용 후 확보한 총 16,522개의 바운딩 박스 데이터셋을 YOLOv5s 모델에 학습하여 7개의 객체를 검출할 수 있도록 함 (윗면, 아랫면, 옆면, 정면, 뒷면, 오류, 손)
* 모델 성능 테스트 결과는 다음과 같음
  *  Precision : 0.982, Recall : 0.986, mAP50 : 0.988
<p align = 'center'><img src = 'https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/35742e3a-9d50-4042-a63c-040edd3ddefe' width = 60% height = 60%></p>

* 빠른 연산을 위해 모델을 onnx 형식으로 변환하여 추론에 사용

#### **B.2.5. 행동 인식을 위한 LSTM 모델 개발**
* 분할한 영상에 Mediapipe를 적용하여 손의 랜드마크를 검출하고 마디간 각도값 및 마디별 좌표값을 계산하여 데이터 프레임 형태로 확보
* 이 때, 손의 랜드마크는 YOLO가 검출한 손의 Bounding Box 안에서만 확인할 수 있도록 함
* 본 프로젝트에서는 총 2개의 LSTM 모델을 개발하였으며 각각 학습한 데이터가 다름
  1. [내려놓기] 동작 시 손의 하강과 상승을 구분할 수 있는 모델 ← 마디별 좌표값 데이터
  2. [회전하기] 동작 시 손의 회전 여부를 확인할 수 있는 모델 ← 마디간 각도값 데이터
* 개발한 LSTM 모델의 성능은 다음과 같음 (테스트 영상 포함)
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
