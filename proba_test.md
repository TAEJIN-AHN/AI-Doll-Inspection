# Proba Test
## 본 테스트의 목적
* 손동작을 구별하는 LSTM 모델이 물건을 내려놓거나 회전하는 동작을 오차없이 0 혹은 1의 확률로만 구분할 수 있다면 이상적이나, 실제로는 그렇지 못함
* 본 테스트에서는 손 동작을 구분하는 LSTM 분류 모델의 Probability Threshold를 설정하여 아래의 오류를 줄이고자 함
   * 내려놓거나 회전하는 손 동작이지만 LSTM 모델이 그렇지 않다고 분류할 경우
   * 내려놓거나 회전하는 손 동작이 아니지만 LSTM 모델이 그렇다고 분류할 경우

<p align = center><img src = 'https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/e0ad6bb3-05dc-40a1-9afa-4173f0e95e70' width = 40% height = 40%></p>

## 1번 테스트
* 295 ~ 304번째 프레임의 각도값을 LSTM 모델에 대입한 결과, 약 42%의 확률로 회전 동작을 분류함
* 301번째 프레임부터 회전 동작을 확인할 수 있으며, 10프레임 중 5개의 프레임(300~304)에 회전 동작이 포함됨
* 바로 다음인 305번째 확률값이 1로 급상승하는 것을 볼 때, 10개 프레임 중 회전 동작이 포함된 프레임의 비율이 높아지면서 확률값도 높아지고 그 비율이 50% 정도를 넘어셔면 확률값이 1에 가까워 짐을 확인할 수 있음
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
* 498 ~ 507번째 프레임의 각도값을 LSTM 모델에 대입한 결과, 약 33%의 확률로 회전 동작을 분류함
* 499번째 프레임부터 회전 동작을 확인할 수 있으며, 10프레임 중 5개의 프레임(499~503)에 회전 동작이 포함됨
* 바로 이전인 506번째 확률값인 100% 에서 33%로 급하락하는 것을 볼 때, 10개 프레임 중 회전 동작이 포함된 프레임의 비율이 낮아지며 확률값이 낮아짐을 알 수 있음
* 1번 테스트와 2번 테스트 전체 프레임 중 회전 동작을 담은 프레임의 비율이 대략 50% 정도로 유사함
* 정지 동작에서 회전 동작으로 변경하는 1번 테스트의 경우 확률값이 42%인 반면 회전 동작에서 정지 동작으로 변경하는 2번 테스트는 33% 정도로 9~10% 가량의 큰 차이를 보임
  
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
* 355 ~ 364번째 프레임의 각도값을 LSTM 모델에 대입한 결과, 약 31%의 확률로 회전 동작을 분류함
* 362번째 프레임부터 회전 동작을 확인할 수 있으며, 10프레임 중 3개의 프레임(362~364)에 회전 동작이 포함됨
* 1번 테스트와 동일하게 정지 동작에서 회전 동작으로 변경하는 과정을 담고 있으나, 10프레임 중 3개만이 회전 동작을 담고 있음
* Threshold를 30% 초반으로 설정한다면, 3번 테스트 영상과 같이 회전 동작이 거의 없는 구간에서도 회전 동작을 분류할 가능성이 있음

<table>
   <tr>
    <td><p align = 'center'>Scatter Plot</p></td>
    <td><p align = 'center'>직전 10프레임 영상</p></td>
   </tr>
  <tr>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/00a99df7-e305-4ab9-a345-0a9a767a5d3d" alt="1" width = 100% height = 100%></p></td>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/f3fd9b6f-059d-478e-913b-6291c96f07c0" alt="2" width = 80% height = 80%></p></td>
  </tr> 
</table>

## 4번 테스트
* 355 ~ 364번째 프레임의 각도값을 LSTM 모델에 대입한 결과, 약 31%의 확률로 회전 동작을 분류함
* 362번째 프레임부터 회전 동작을 확인할 수 있으며, 10프레임 중 3개의 프레임(362~364)에 회전 동작이 포함됨
* 1번 테스트와 동일하게 정지 동작에서 회전 동작으로 변경하는 과정을 담고 있으나, 10프레임 중 3개만이 회전 동작을 담고 있음
* Threshold를 30% 초반으로 설정한다면, 3번 테스트 영상과 같이 회전 동작이 거의 없는 구간에서도 회전 동작을 분류할 가능성이 있음

<table>
   <tr>
    <td><p align = 'center'>Scatter Plot</p></td>
    <td><p align = 'center'>직전 10프레임 영상</p></td>
   </tr>
  <tr>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/00a99df7-e305-4ab9-a345-0a9a767a5d3d" alt="1" width = 100% height = 100%></p></td>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/f3fd9b6f-059d-478e-913b-6291c96f07c0" alt="2" width = 80% height = 80%></p></td>
  </tr> 
</table>

## 결론
* 정지 동작에서 회전 동작으로 변경하는 구간과 회전 동작에서 정지 동작으로 변경하는 구간의 확률값 차이 (42%, 33%)가 대략 9~10% 정도로 큼
* Threshold를 30% 초반으로 설정한다면, 3번 테스트 영상과 같이 회전 동작이 거의 없는 구간에서도 회전 동작을 분류할 가능성이 있음
* Threshold를 40% 중~후반 이후로 설정한다면, 회전 동작이 충분히 포함되어 있는 구간도 회전 동작으로 분류하지 않을 가능성이 있음 
* Probability Threshold를 35% ~ 40% 정도로 설정할 시, 조금 더 안정적으로 분류 오류를 줄일 수 있다고 예상됨
