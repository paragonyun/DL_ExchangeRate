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

<br>

<br>

## 📃 파일 설명
`Normal_Seq2Seq` : 성능 비교를 위한 가장 기본적인 형태의 Seq2Seq 모델입니다. 

`Attention_Seq2Seq` : 기본적인 Seq2Seq에 Attention이 적용된 모델입니다.  

`Event_Seq2Seq` : Event Attention이 적용된 모델입니다.  


<br>

<br>


## ✨ Training Methodology
학습 방식의 차이보다 기본적인 모델의 구조차이를 확인하는 것이 목표였기 때문에, 학습 방식에 대한 튜닝은 그렇게까지 세부적으로 진행하지는 않았습니다.
- `Epoch` : 3000
- `Optimizer` : Adam
- `Larning Rate` : 0.01 
- `Scheduler` : None