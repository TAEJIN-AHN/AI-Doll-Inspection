# Proba Test
## 본 테스트의 목적
* 손동작을 구별하는 LSTM 모델이 회전과 그 외의 동작을 오차없이 0 혹은 1의 확률로 구분한다면 이상적이나, 실제로는 그렇지 못함
* 본 테스트에서는 손 동작을 구분하는 LSTM 분류 모델의 Probabilty Threshold를 설정하여 아래의 오류를 줄이고자 함
   * 회전 동작이지만 LSTM 모델이 그렇지 않다고 분류할 경우
   * 회전 동작이 아니지만 LSTM 모델이 회전동작이라고 분류할 경우
<p align = 'center'>
  <img src = 'https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/d36ca409-54e5-4e56-912f-c071eb84af19'>
</p>
  
<table>
   <tr>
    <td><p align = 'center'>Scatter Plot</p></td>
    <td><p align = 'center'>직전 10프레임 영상</p></td>
   </tr>
  <tr>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/787d42a5-9ae4-426b-a0a1-0e971b78fec9" alt="1" width = 80% height = 80%></p></td>
   <td><p align = 'center'><img src="https://github.com/TAEJIN-AHN/AI-Doll-Inspection/assets/125945387/53d203d0-65e3-46a4-a28a-675dd4102d85" alt="2" width = 80% height = 80%></p></td>
  </tr> 
</table>

