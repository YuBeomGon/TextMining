# TextMining

영화 리뷰에 대한 분류 모델로서 데이터는 데이터셋은 Naver sentiment movie corpus를 사용.
1. https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html를 따라 간단한 분류모델 테스트
2. https://ratsgo.github.io/natural%20language%20processing/2017/03/19/CNN/  코드를 베이스로 하여 약간의 수정을 더한 후 분류 모델 training 및 test

3. 2번 테스트 시  Failed to get convolution algorithm. This is probably because cuDNN failed to initialize 와 같은 에러가 나올 수 있다. 
   Gpu memory 관련된 사항으로 batch size를 줄이거나 전체 parameter의 숫자를 줄이도록 하자.
