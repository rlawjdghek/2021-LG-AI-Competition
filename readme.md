### 2021년 8월 8일) 데이콘: 카메라 이미지 품질 향상 - 슬기로운LG생활 팀

#### 제출 파일
* readme.md 
* smart_lg_life_team_docker.tar
* 슬기로운LG생활.pptx


#### System Requirements
* docker = 20.10.6
* nvidia-driver = 460.39
* cuda 11.1
* Ubuntu 18.04
* python 3.8.8 (도커내에 이미 설치가 되어있습니다.) 
* pytorch 1.8.1 (도커내에 이미 설치가 되어있습니다.) 
* numpy 1.19.2 (도커내에 이미 설치가 되어있습니다.)
* opencv 4.5.3.56 (도커내에 이미 설치가 되어있습니다.)
* matplotlib 3.4.2 (도커내에 이미 설치가 되어있습니다.) 
* segmentation-models-pytorch (도커내에 이미 설치가 되어있습니다.)

* pytorch-polynomial-lr-decay (도커내에 이미 설치가 되어있습니다.) 
```bash
pip install git+https://github.com/cmpark0126/pytorch-polynomial-lr-decay.git
```


#### 훈련 코드 실행
1. 도커 이미지 생성
```bash
docker load -i smart_lg_life_docker.tar
```

2. 도커 컨테이너 생성 및 실행
```bash
docker run -it --gpus all --ipc=host --name smart_lg_life_team smart_lg_life:final
```

3. 훈련 코드 실행
```bash
cd /workspace/smart_lg_life_team/src
bash training.sh
```

#### 테스트 코드 실행
1. 도커 내에서 테스트 코드 실행
```bash
cd /workspace/smart_lg_life_team/src
bash inference.sh
```

