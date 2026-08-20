# RNN(Recurrent Neural Network)

## Feedforward 신경망

<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/1.png" width="500" height="250"/>
- 흐름이 단방향인 신경망
- 시계열의 데이터를 학습하기 어려움

## RNN의 구조


<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/17.png" width="600" height="250"/>

- one to one : 기본 Neural Networks
- one to many : 이미지에 설명을 다는 캡셔닝, image → sequence word 출력
- many to one : 감정분석, sequence of words → sentiment
- many to many : 번역기

<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/18.png" width="600" height="250"/>

위에서 봤던 구조 압축 ↓

<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/2.png" width="500" height="300"/>

- **hidden node가 방향을 가진 edge로 연결돼 순환구조를 이루는 인공신경망**
- input $Xt$
, output $yt$ , 현재 상태의 hidden state  $h_{t}$는 직전 시점의 hidden state $h_{t-1}$를 받아 갱신
- hidden state의 활성함수는 비선형 함수인 **하이퍼볼릭탄젠트(tanh)**
- 시퀀스 길이에 관계없이 인풋과 아웃풋을 받아들일 수 있는 네트워크 구조
- 과거의 정보($h_{t-1}$)를 기억하는 동시에 최신 데이터로 갱신될 수 있음



## RNN의 순전파

1. 순환 신경망의 기본 구조로 x값들을 Input으로 넣고, 중간에 hidden state가 있고 output은 y값

<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/4.png" width="500" height="250"/>




2. 두가지의 W가중치 고려하기
- input이 들어올때 $W_{xh}$와 곱해진다.
- 과거의 값이 들어올때 $W_{hh}$와 곱해진다.
- 중요한점 : $**W_{xh}$와 $W_{hh}$가중치는 전부 다 같은 값을 쓴다.** (→ 나중에 gradient descent 문제..)

<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/3.png" width="500" height="250"/>




3. x값과 곱한 w들을 더해주고, 편향 b까지 더해주기

<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/5.png" width="500" height="250"/>



4. 활성화함수인 tanh 적용
- 은닉층 : $h_{t} = tanh(x_{1} W_{xh} + h_{t-1} W_{hh} + b)$

<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/6.png" width="500" height="250"/>



5. 최종 출력 y값
- 출력층 : $y_{t} = W_{y} h_{t} + b$

<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/7.png" width="500" height="250"/>





## RNN의 역전파(BackPropagation Through Time)


### 1. 덧셈 노드의 역전파

<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/15.jpg" width="500" height="250"/>
z = x + y 

위의 임의의 계산으로 가는 노드는 x를 기준으로 편미분을 한다. → y가 사라지고 1이 된다.

아래의 임의의 계산으로 가는 노드는 y를 기준으로 편미분을 한다. → x가 사라지고 1이 된다.

→ + 노드를 대상으로 하는 역전파는 입력받은 데이터를 그대로 흘러준다.

### 2. 곱셈 노드의 역전파

<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/16.jpg" width="500" height="350"/>

z = xy

x로 편미분 : y

y로 편미분 : x

→ 곱하기 노드에서는 순전파 때의 입력 신호들에 서로 바꾼 값을 곱하여 하류로 보낸다.

## LSTM

- RNN의 문제점 : **Vanishing Gradient**

> ex ) Tom is watching TV in his room, Mary came into the room, Mary said hi to ?
> 

<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/9.png" width="500" height="350"/>
정답 레이블이 Tom일때, “Tom이 방에서 티비를 보고 있음”과 “Mary가 방에 들어옴" 이라는 정보를 기억해야 한다. → RNN 은닉층에 인코딩해서 보관

1. Neural Network는 오류를 최소화 하기 위해서 가중치 업데이트, 과거의 방향으로 기울기를 전달. 
2. RNN은 출력층 바로 밑에 있는 뉴런이 아니라, 위의 예시처럼 모든 뉴런이 훨씬 오래 전에 기여 → 이 뉴런들에게 시간을 거슬러 다시 전파해야 함

### **기울기 소실 / 기울기 폭발 원인**

### 1. tanh 활성화함수 적용하는 경우

<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/10.png" width="500" height="250"/>
신경망은 **곱하기 연산**을 기반으로 만들어져 있기 때문에, 미분값이 1보다 조금만 커도 gradient값이 **발산**하게 되고, 1보다 조금만 작아도 gradient값이 소실하게 됨.

> ex) 1.1 ^ 100 = 13,780.6
0.9 ^ 100 = 0.0000265
> 

tanh의 미분함수 → 0과 1사이, Backpropagation시 미분값이 소실될 가능성이 있음

### 2. 행렬곱 노드를 지나는 경우

Backpropagation시 $\ast W_{h}$ 행렬곱 연산을 하게 됨

매번 똑같은 가중치를 여러 번 곱할때 1보다 작은 숫자라면 0으로 **수렴**하고, 1보다 크다면 **발산.**

### ✔️ 해결책 : LSTM

<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/11.png" width="500" height="250"/>
RNN의 반복 모듈이 하나의 layer를 갖고 있는 표준적인 모습이다.

<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/12.png" width="500" height="250"/>
LSTM의 반복 모듈에는 4개의 상호작용하는 layer가 들어있다.






## LSTM의 특징

## 1. Gates

4개의 모듈은 3개의 Gate로 표현된다.

<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/14.png" width="500" height="250"/>

- forget gate(f) : 과거 정보를 잊기위한 게이트
- input gate(i) : 현재 정보를 기억하기 위한 게이트
- output gate(o) : 최종 결과 h를 위한 게이트

모든 gate는 sigmoid 함수를 사용하고 출력 범위는 0~1이기 때문에 그 값이 0이라면 정보를 잊고, 1이라면 정보를 온전히 기억하게 된다.


## 2. Cell State

<img src="https://github.com/eunbinni/TIL/blob/main/RNN/images/13.png" width="500" height="250"/>

Cell State : 맨 위를 가로지르는 선

RNN은 곱하기 연산으로만 이루어져 있다.

LSTM에서는 더하기 기호(+)를 사용함으로써 vanishing gradient 방지
