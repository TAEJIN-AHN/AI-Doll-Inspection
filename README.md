![타이틀](https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/c9dbff59-f53a-4222-a0fd-9f4d772e5cb3)

## **A. 목차**
  * A. TOC
  * B. C2C 플랫폼을 위한 불량 제품 검수 모델 개발
    * B.1. 문제정의
    * B.2. 프로젝트 진행(주요 액션)
      * B.2.1. 학습 및 테스트용 동영상 촬영
      * B.2.2. 어노테이션 (Bounding Box)
      * B.2.3. 객체 인식을 위한 YOLOv5s 모델 개발
      * B.2.4. 행동 인식을 위한 LSTM 모델 개발
      * B.2.5. YOLO와 LSTM 모델을 이식한 검수 절차 구축
      * B.2.6. PyQt를 활용한 검수 PC 어플리케이션 개발
    * B.3. 결과
  * C. Deck
  * D. Methods Used
  * E. Contributin Members
--- 
## **B. C2C 플랫폼을 위한 불량 제품 검수 모델 개발**

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
    <td><p align = 'center'>head</p></td>
    <td><p align = 'center'>rotate</p></td>
    <td><p align = 'center'>bottom</p></td>
    <td><p align = 'center'>put</p></td>
   </tr>
  <tr>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/b72733b2-c3df-4e42-80aa-e59efd854d37" alt="1" width = 80% height = 80%></p></td>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/53d203d0-65e3-46a4-a28a-675dd4102d85" alt="2" width = 80% height = 80%></p></td>     
   <td><img src="./Scshot/cab_arrived.png" alt="3" width = 360px height = 640px></td>
   <td><img src="./Scshot/trip_end.png" alt="4" width = 360px height = 640px></td>
  </tr> 
https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/53d203d0-65e3-46a4-a28a-675dd4102d85
</table>

#### **B.2.3. 어노테이션 (Bounding Box)**
#### **B.2.4. 객체 인식을 위한 YOLOv5s 모델 개발**
#### **B.2.5. 행동 인식을 위한 LSTM 모델 개발**
#### **B.2.6. YOLO와 LSTM 모델을 이식한 검수 절차 구축**
#### **B.2.7. PyQt를 활용한 검수 PC 어플리케이션 개발**
