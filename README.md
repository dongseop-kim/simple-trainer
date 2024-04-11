# Train deeplearning models

## 1. 환경 세팅
host는 linux 환경을 기준으로 설명합니다. gpu 사용시 driver version >= 510.183 이 필요합니다. 

```bash
    $ git clone https://github.com/dongseop-kim/simple-trainer.git
    $ cd simple-trainer
    $ docker build -t simple-trainer:v1.0 -f ./docker/Dockerfile .
    $ docker run -it --rm --gpus all simple-trainer:v1.0 /bin/bash
    # in container
    $ python3 trainer/train_all.py --config ./trainer/configs/train_mnist.yaml # sample test
```


## 2. training
전체적인 과정은 학습을 위한 config file 작성 및 학습을 위한 모델, 엔진 구현이 있습니다.
기존의 학습 세팅이 갖춰진 경우 config만 수정하여 학습을 빠르게 진행할 수 있습니다.

### 2.1 training config 작성

trainer/configs/train_mnist.yaml 의 save_dir, data_dir 등을 수정합니다. 그 외 batch, transform 등을 수정할 수 있습니다. config 파일 작성이 끝났다면 아래의 명령어를 실행합니다.

```bash
    $ cd trainer
    # e.g. python3 train_all.py --config configs/train_mnist.yaml
    $ python3 train_all.py --config config/to/your/config.yaml
```

### 2.2 training model 구현

현 repo에서는 모델은 크게 encoder, decoder, header로 구분 됩니다. encoder는 timm 패키지를 적극 사용하며, decoder 및 header는 각자 task에 맞춰 구현합니다. 

### 2.3 export trained model
onnx로 exporting 

```bash
    $ python3 export.py --help
    $ python3 export.py --config config/to/your/config.yaml --weight path/to/your/weightfile --output path/to/your/onnxfile
```

### 2.4 model registration
학습된 모델을 저장하고, 추후에 사용하기 위해 모델을 등록합니다. 

```bash
    $ mlflow server --backend-store-uri ./mlruns --default-artifact-root ./mlartifacts
    $ python3 /opt/simple-trainer/register.py --experiment-id 0 --model-name test-01 --model-path path/to/your/onnxfile
```

## 3. predict
predict image with registered model

```bash
    # e.g. python3 predict.py --image ./assets/test_7.png
    $ python3 predict.py --image path/to/your/image

    >>>
            ㅁ ㅁ ㅁ ㅁ ㅁ ㅁ ㅁ ㅁ ㅁ ㅁ ㅁ                
            ㅁ ㅁ ㅁ ㅁ ㅁ ㅁ ㅁ ㅁ ㅁ ㅁ ㅁ                
                        ㅁ ㅁ ㅁ ㅁ ㅁ ㅁ ㅁ ㅁ             
                                    ㅁ ㅁ ㅁ                
                                    ㅁ ㅁ ㅁ                
                                    ㅁ ㅁ                   
                                 ㅁ ㅁ ㅁ                   
                              ㅁ ㅁ ㅁ                      
                              ㅁ ㅁ                         
                           ㅁ ㅁ ㅁ                         
                        ㅁ ㅁ ㅁ                            
                        ㅁ ㅁ ㅁ                            
                     ㅁ ㅁ ㅁ                               
                     ㅁ ㅁ ㅁ                               
                     ㅁ ㅁ ㅁ                               
        
        Predicted class: 7
```



