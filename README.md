# Simple Trainer for Model Training

### 1. Setting up the environment

```bash
    $ git clone https://github.com/dongseop-kim/simple-trainer.git
    $ cd simple-trainer
    $ docker build -t simple-trainer:v1.0 -f ./docker/Dockerfile .
    $ docker run -it --rm --gpus all simple-trainer:v1.0 /bin/bash
```

---

### 2. Train Model
전체적인 과정은 학습을 위한 config file 작성 및 학습을 위한 모델, 엔진 구현이 있습니다.
기존의 학습 세팅이 갖춰진 경우 config만 수정하여 학습을 빠르게 진행할 수 있습니다.


#### 2.1 Writing training configuration

trainer/configs/train_mnist.yaml 의 save_dir, data_dir 등을 수정합니다. 그 외 batch, transform 등을 수정할 수 있습니다. config 파일 작성이 끝났다면 아래의 명령어를 실행합니다.

```bash
    $ cd trainer
    # e.g. python3 train_all.py --config configs/train_mnist.yaml
    $ python3 train_all.py --config config/to/your/config.yaml
```
