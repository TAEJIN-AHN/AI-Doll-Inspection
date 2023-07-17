# Proba Test
## 본 테스트의 목적
* 손동작을 구별하는 LSTM 모델이 회전과 그 외의 동작을 오차없이 0 혹은 1의 확률로만 구분한다면 이상적이나, 실제로는 그렇지 못함
* 본 테스트에서는 손 동작을 구분하는 LSTM 분류 모델의 Probabilty Threshold를 설정하여 아래의 오류를 줄이고자 함
   * 회전 동작이지만 LSTM 모델이 그렇지 않다고 분류할 경우
   * 회전 동작이 아니지만 LSTM 모델이 회전동작이라고 분류할 경우
<p align = 'center'>
  <img src = 'https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/97f19243-f466-424f-bcce-90663862e33b' width = 40% height = 40%>
</p>

## 1번 테스트
* 295 ~ 304번째 프레임의 각도값을 LSTM 모델에 대입한 결과, 약 42%의 확률로 회전 동작을 분류하였다.
* 301번째 프레임부터 회전 동작을 확인할 수 있으며, 10프레임 중 5개의 프레임(300~304)에 회전 동작이 포함되어 있다.
* 바로 다음인 305번째 확률값이 1로 급상승하는 것을 볼 때, 10개 프레임 중 회전 동작이 포함된 프레임의 비율이 높아지면서 확률값도 높아짐을 확인할 수 있다

<table>
   <tr>
    <td><p align = 'center'>Scatter Plot</p></td>
    <td><p align = 'center'>직전 10프레임 영상</p></td>
   </tr>
  <tr>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/787d42a5-9ae4-426b-a0a1-0e971b78fec9" alt="1" width = 100% height = 100%></p></td>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/273c3729-7c55-449e-be05-07e0753405f8" alt="2" width = 80% height = 80%></p></td>
  </tr> 
</table>

## 2번 테스트
* 498 ~ 507번째 프레임의 각도값을 LSTM 모델에 대입한 결과, 약 33%의 확률로 Rotate를 분류하였다.
* 499번째 프레임부터 회전 동작을 확인할 수 있으며, 10프레임 중 5개의 프레임(499~503)에 회전 동작이 포함되어 있다.
* 바로 이전인 506번째 확률값인 100% 에서 33%로 급하락하는 것을 볼 때, 10개 프레임 중 회전 동작이 포함된 프레임의 비율이 낮아지며 확률값이 낮아짐을 알 수 있다.
* 1번 테스트와 2번 테스트 전체 프레임 중 회전 동작을 담은 프레임의 비율이 대략 50% 정도로 유사하다.
* 정지 동작에서 회전 동작으로 변경하는 1번 테스트의 경우 확률값이 42%인 반면 회전 동작에서 정지 동작으로 변경하는 2번 테스트는 33% 정도로 9~10% 가량의 큰 차이를 보인다
  
<table>
   <tr>
    <td><p align = 'center'>Scatter Plot</p></td>
    <td><p align = 'center'>직전 10프레임 영상</p></td>
   </tr>
  <tr>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/28dd95b7-a4eb-406d-ac1d-98e78b00acd5" alt="1" width = 100% height = 100%></p></td>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/29e6ecd6-61ae-439d-96be-cf99879c2be6" alt="2" width = 80% height = 80%></p></td>
  </tr> 
</table>

## 3번 테스트
* 498 ~ 507번째 프레임의 각도값을 LSTM 모델에 대입한 결과, 약 33%의 확률로 Rotate를 분류하였다.
* 499번째 프레임부터 회전 동작을 확인할 수 있으며, 10프레임 중 5개의 프레임(499~503)에 회전 동작이 포함되어 있다.
* 바로 이전인 506번째 확률값인 100% 에서 33%로 급하락하는 것을 볼 때, 10개 프레임 중 회전 동작이 포함된 프레임의 비율이 낮아지며 확률값이 낮아짐을 알 수 있다.
* 1번 테스트와 2번 테스트 전체 프레임 중 회전 동작을 담은 프레임의 비율이 대략 50% 정도로 유사하다.
* 정지 동작에서 회전 동작으로 변경하는 1번 테스트의 경우 확률값이 42%인 반면 회전 동작에서 정지 동작으로 변경하는 2번 테스트는 33% 정도로 9~10% 가량의 큰 차이를 보인다

<table>
   <tr>
    <td><p align = 'center'>Scatter Plot</p></td>
    <td><p align = 'center'>직전 10프레임 영상</p></td>
   </tr>
  <tr>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/00a99df7-e305-4ab9-a345-0a9a767a5d3d" alt="1" width = 100% height = 100%></p></td>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/29e6ecd6-61ae-439d-96be-cf99879c2be6" alt="2" width = 80% height = 80%></p></td>
  </tr> 
</table>
