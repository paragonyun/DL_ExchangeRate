# DL_ExchangeRate
📈 딥러닝을 활용한 단기 환율 예측 (논문 작성용)  
📈 Prediction of Exchange Rate with Deep Learning (for paper) 

<br>

## 🚩 방향성
> 1. 기본적으로 2주치를 보고 앞으로의 일주일치를 예상하는 모델
>
> 2. 정확한 비교를 위해 모델의 구조 외의 모든 조건을 동일하게 설정.
> 
> 3. 기본적인 형태부터 Attention, Event Attention을 적용해나가는 과정을 담음.

> Event Attention: 매 예측 step 마다, 과거 impact가 컸던 event와의 유사도를 비교하여 해당 정보를 반영할 수 있는 모델.
> 과거의 traumatic한 정보를 고려한다는 아이디어에서 착안
> 예측을 진행할 때, Input Sequence의 Temporal한 정보를 반영함과 동시에 과거의 특정 시점과의 비교를 동시에 진행한다는 것이 차이점.

<br>

<br>

## 📃 파일 설명
`Normal_Seq2Seq` : 성능 비교를 위한 가장 기본적인 형태의 Seq2Seq 모델입니다. 

`Attention_Seq2Seq` : 기본적인 Seq2Seq에 Attention이 적용된 모델입니다.  

`Event_Seq2Seq` : Event Attention이 적용된 모델입니다.  

`Sep_Prediction` : 모델이 급격한 변화에도 강건한지 확인하기 위한 코드입니다. `argparser`로 모델 테스팅을 더 간결화 시켰습니다



<br>

## TODO
- Validation
- Compare with "Real Input Sequence" not Context Vector
- Change num_layers, to GRU etc... 
- with WandB

<br>


## ✨ Training Methodology
학습 방식의 차이보다 기본적인 모델의 구조차이를 확인하는 것이 목표였기 때문에, 학습 방식에 대한 튜닝은 그렇게까지 세부적으로 진행하지는 않았습니다.
- `Epoch` : 1000
- `Optimizer` : Adam
- `Larning Rate` : 0.001 
- `Scheduler` : None
