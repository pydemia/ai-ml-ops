# Distributed Training-`tensorflow:2.x`

[Official Docs](https://www.tensorflow.org/guide/distributed_training)
[Templates](https://github.com/tensorflow/ecosystem)

## Table of Contents

* 용어 설명
* Code 및 API 정리
  * 
* 
  


## Overview

* `tf.distribute`: 분산 작업(학습/추론)을 위한 모듈

* `tf.distribute.Strategy`: 작업 시에 알고리즘을 Multi Device/Machine에 분산시키기 위한 실행 전략으로서의 API

목표
* 사용하기 쉽고, 연구원, 기계 학습 엔지니어 등 여러 사용자 층을 지원할 것.
* 그대로 적용하기만 하면 좋은 성능을 보일 것.
* 전략들을 쉽게 갈아 끼울 수 있을 것.

`tf.keras` & `tf.estimator`와 함께 사용 가능. 사용자 정의 loop용 API 지원.
2.0에서는 eager execution 및 `tf.function`으로 graph 실행 가능.

약간의 코드 수정이 필요하지만
Tensorflow 구성요소들인
- 변수(variable)
- 층(layer)
- 모델(model)
- 옵티마이저(optimizer)
- 지표(metric)
- 서머리(summary)
- 체크포인트(checkpoint, ckpt)

들을 전략(Strategy)으로 이해하고 처리할 수 있도록 작성해 놓아서 변경이 편리함.


---
## 용어사전

| 용어 | 내용 |
| :-- | :--- |
| `machine` | 장비. 보통 VM. |
| `device` | CPU, GPU, TPU 등. `tensorflow`가 돌아가는 장비의 부분.<br>(ex. 1 VM에 2 GPU -> 1 장비에 2 device) |
| `replica` | 한 Input Slice를 돌리는 모델 사본 하나.<br>지금은 `모델 병렬화`가 구현되지 않아서 1 `worker device`에서만 돌아감.<br>model parallelism이 구현되면 복수의 `worker device` 위에 존재 가능.
| `worker` | 복제된 계산이 돌아가는 `물리적 device`(CPU, TPU)를 담고 있는 `물리적 장비`. 1 `worker`는 1개 이상의 `replica`를 갖고 있음. |
| 



- `worker device`: 계산하는 device.
  - `parameter device`: _variable_ 을 보관하는 device.
  - `tf.distribute.MirroredStrategy`: `worker device` == `parameter device`
  - `tf.distribute.experimental.CentralStorageStrategy`: 1 `parameter device`를 가진다.(`worker device`이거나 `CPU`)
  - `tf.distribute.experimental.ParameterServerStrategy`: `parameter server`를 별도로 두어 _variable_ 보관. |
| 
* `machine`: 장비. 보통 VM.
* `device`: CPU, GPU, TPU 등. `tensorflow`가 돌아가는 machine의 부분.
  (ex. 1 VM에 2 GPU -> 1 장비에 2 device)
  - `worker device`: 계산하는 device.
  - `parameter device`: _variable_ 을 보관하는 device.
  - `tf.distribute.MirroredStrategy`: `worker device` == `parameter device`
  - `tf.distribute.experimental.CentralStorageStrategy`: 1 `parameter device`를 가진다.(`worker device`이거나 `CPU`)
  - `tf.distribute.experimental.ParameterServerStrategy`: `parameter server`를 별도로 두어 _variable_ 보관.
* `replica`: 한 Input Slice를 돌리는 모델 사본 하나. 지금은 `모델 병렬화`가 구현되지 않아서 1 `worker device`에서만 돌아감. model parallelism이 구현되면 복수의 `worker device` 위에 존재 가능.
* `worker`: 복제된 계산이 돌아가는 `물리적 device`(CPU, TPU)를 담고 있는 `물리적 장비`. 1 `worker`는 1개 이상의 `replica`를 갖고 있음.
  보통 1 `worker`는 1 장비에 대응하지만, `모델 병렬화`가 적용된 큰 모델에서는 1 `worker`가 2개 이상의 장비 위에 존재할 수 있음.
  대개 1 `worker`마다 1 `input pipeline`을 붙여서, 이 `worker`에 속한 모든 `replica`에 데이터를 feeding함.
* `데이터 병렬화(Data Parallelism)`: 복수의 모델 사본을 다른 Input Slice에 적용하는 것.
* `모델 병렬화(Model Parallelism)`: 단일 모델 사본을 복수의 `device`에 적용하는 것.(향후 지원 예정)
* `host`: `worker device`가 있는 장비의 `CPU device`. 보통 `input pipeline`을 돌리기 위해 사용.
* `synchronous training` or `sync training`: 모델 _variable_ 업데이트 전, 각각 `replica`를 취합하는 부분. 각 `replica`가 독립적으로 모델 _variable_ 을 업데이트하는 `async training`과는 정 반대이며, `replica`를 그룹으로 partitioning해서 그룹 내는 `sync`, 그룹끼리는 독립적으로(`async`) 구성할 수도 있다.
* `parameter server`: _parameter_ 와 _variable_ 사본을 보관하는 1개 이상의 장비.  `tf.distribute.experimental.ParameterServerStrategy`에서 쓰임.  
  각 `replica`는 1 step 시작할 때 _variable_ 을 받아 와서, step 끝에 업데이트된 값으로 새로 보내게 됨(`sync`/`async`에 따라 달라지며, 지금은 `async`만 지원).
  - `tf.distribute.experimental.CentralStorageStrategy`는 모든 _variable_ 을 같은 장비에 있는 단일 `device`에 모아 `sync training` 수행.
  - `tf.distribute.MirroredStrategy`는 _variable_ 을 그냥 mirroring하여 복수의 `device`에 보냄.
* `mirrored variable`: 복수의 `device`에 복사된 _variable_. `sync training` 적용.
* `reductions` & `all-reduce` : 복수의 `device`에 있는 값 aggregating 방법. `sync training`에 적용.


---
## 전략(Strategy)

* 훈련방식: `sync`/`async` training
  - `sync`: `mirrored variable` & `all-reduce`로 구현
  - `async`: `parameter server`로 구현
* 분산방식: 1 


| 훈련 API | MirroredStrategy | TPUStrategy | MultiWorkerMirroredStrategy | CentralStorageStrategy | ParameterServerStrategy |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Keras API | 지원(fully) | 2.0 RC 지원 예정 | 지원(experimental) | 지원(experimental) | 2.0 이후 지원 예정 |
사용자 정의 훈련 루프 | 지원(experimental) | 지원(experimental) | 2.0 이후 지원 예정 | 2.0 RC 지원 예정 | 미지원 |
Estimator API | 지원(limited) | 지원(limited) | 지원(limited) | 지원(limited) | 지원(limited) |

---
### MirroredStrategy

단일 장비, 다중 GPU 동기 분산 training.
Mirroring 방식.

1. 각 GPU(`device`)마다 `replica` 생성
2. 모델 variable 생성(가상 variable=`MirroredVariable`)
3. 각 `replica`에 모델 variable mirroring
4. 각 `replica`가 모델 학습-loss 계산
5. `all-reduce` 알고리즘으로 가상 variable 업데이트
6. 다시 각 `replica`에 variable mirroring(3번으로)


```py
# use All GPUs in a machine
mirrored_strategy = tf.distribute.MirroredStrategy(
    devices=None,           # device(GPU) 선택.
    cross_device_ops=None,  # 장치 간 통신 방법 변경. 기본값: tf.distribute.NcclAllReduce
)
# use some GPUS in the machine
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
```

---
### CentralStorageStrategy

단일 장비, 다중 GPU 동기 분산 training.
하지만 variable mirroring 하지 않고, CPU에서 관리.
작업은 모든 local GPU(`device`)로 복제됨.

```py
central_storage_strategy = tf.distribute.experimental.CentralStorageStrategy()
```

---
### MultiWorkerMirroredStrategy

__MirroredStrategy__ 와 비슷하지만, 복수의 `worker` 사용한 동기 분산 training.
Multi `worker` x Multi GPU(`device`)

다중 `worker` 간 통신은 `tf.distribute.CollectiveOps` 사용하여
variable들을 같은 값으로 유지.

```py
multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    communication=tf.distribute.experimental.CollectiveCommunication.AUTO,
    cluster_resolver=None,
)

# CollectiveCommunication.RING: gRPC를 사용한 링 네트워크 기반의 수집 연산
# CollectiveCommunication.NCCL: Nvidia의 NCCL을 사용하여 수집 연산을 구현
```

#### `TF_CONFIG` 변수

다중 `worker` 설정 시 `TF_CONFIG` 환경변수(`json` 형식)로 클러스터 설정.
클러스터를 구성하는 task와, 그 address 및 role 정의.
* [`kubernetes` Template 제공: Training 작업에 맞게 `TF_CONFIG` 설정](https://github.com/tensorflow/ecosystem)

예시: 
3 `worker`

```py
# task type 종류:
# {"chief"(지휘자), "worker"(워커), "ps"(파라미터 서버), "evaluator"(평가자)}
# ps는 `ParameterServerStrategy` 사용 시에만 가능.

os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["host1:port", "host2:port", "host3:port"]
    },
   "task": {"type": "worker", "index": 1}
})
```

---
### TPUStrategy

기본적으로, MirroredStrategy와 동일하지만 연산 효율화 적용.

```py
# 사용 가능한 TPU 탐색("tpu" 매개변수에 지정)
cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
    tpu=tpu_address)
tf.config.experimental_connect_to_host(cluster_resolver.master())

# TPU는 계산 전 초기화 필요. 명시적으로 호출.
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)

tpu_strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
```

---
### ParameterServerStrategy

일부 장비는 `worker`, 일부 장비는 `parameter server` 역할 담당.
모델의 각 variable은 한 `parameter server`에 할당되고 계산 작업은 모든 `worker`의 GPU(`device`)에 복제됨.

```py
ps_strategy = tf.distribute.experimental.ParameterServerStrategy()
```

다중 `worker` 설정을 위해서는, `TF_CONFIG` 설정 필요.

예시: 
3 `worker`, 2 ps(`parameter server`)

```py
# task type 종류:
# {"chief"(지휘자), "worker"(워커), "ps"(파라미터 서버), "evaluator"(평가자)}
# ps는 `ParameterServerStrategy` 사용 시에만 가능.

os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["host1:port", "host2:port", "host3:port"],
        "ps": ["host4:port", "host5:port"]
    },
   "task": {"type": "worker", "index": 1}  # 2번째 worker라는 뜻(0부터)
})
```


---
### 예시 1: `tf.keras`로 `tf.distribute.Strategy` 사용하기

1. 적절한 `tf.distribute.Strategy` Instance(객체) 생성
2. `strategy.scope` 안으로 `tf.keras` 선언 & compile 작업 옮기기
  
```py
# 분산 Strategy 선언
mirrored_strategy = tf.distribute.MirroredStrategy()

# 이 scope 내에서 선언해서, Mirrored Variables 생성.
# 이 scope 내에서 model.compile로 optimizer 선언:
#   이 strategy 적용하여 모델 training한다는 의미
with mirrored_strategy.scope():
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    model.compile(loss='mse', optimizer='sgd')

# Input Pipeline 생성: Dataset
dataset = (
    tf.data.Dataset.from_tensors(([1.], [1.]))
    .repeat(1000)
    .batch(10)
)

# Training: model.fit 호출
model.fit(dataset, epochs=2)

# Inference(Prediction)
model.evaluate(dataset)
```

### 예시 2: `tf.estimator`로 `tf.distribute.Strategy` 사용하기

비동기 `parameter server` 방식 지원하는 Tensorflow 초기 API

예시 1과 다른 점: Strategy Instance(객체)를 `tf.estimator.RunConfig`에 전달하는 방식

```py
# 분산 Strategy 선언
mirrored_strategy = tf.distribute.MirroredStrategy()

# 분산 Strategy을 전달하기 위해 Config 설정
config = tf.estimator.RunConfig(
    train_distribute=mirrored_strategy,  # Training 분산 Strategy
    eval_distribute=mirrored_strategy,   # Inference(Prediction) 분산 Strategy
)

# `tf.estimator` 모델에 config 변수로 전달
regressor = tf.estimator.LinearRegressor(
    feature_columns=[tf.feature_column.numeric_column('feats')],
    optimizer='SGD',
    config=config,
)

def input_fn():
    # Input Pipeline 생성: Dataset
    dataset = tf.data.Dataset.from_tensors(({"feats":[1.]}, [1.]))
    return dataset.repeat(1000).batch(10)

# Training: model.fit 호출
regressor.train(input_fn=input_fn, steps=2)

# Inference(Prediction)
regressor.evaluate(input_fn=input_fn, steps=1)
```

* `tf.estimator`에서는 `input_fn`을 정의하는 형태로 모델에 데이터를 공급함.
  `worker`나 `device`에 어떻게 나눌지도 수동으로 설정해야 함.
  기본적으로 `worker`마다 한번씩 호출되므로 `worker`마다 Dataset 받게 됨.
  예시 1과 같이 동작하게 하려면, 
  전체 batch_size = `PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync`
<br/>

* [`tf.estimator`를 활용하여 다중 `worker` strategy로 `MNIST` classification training(모델 `model_fn`->`tf.keras`)](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator#multiworkermirroredstrategy)
* [`kubernetes` Template 제공: `tf.estimator`를 활용하여 `tf.keras` 모델을 다중 `worker` strategy로 training](https://github.com/tensorflow/ecosystem/tree/master/distribution_strategy)

* [`MirroredStrategy` or `MultiWorkerMirroredStrategy` 적용 가능한 공식 ResNet50 모델](https://github.com/tensorflow/models/blob/master/official/r1/resnet/imagenet_main.py)


### 예시 3: 사용자 정의 loop에 `tf.distribute.Strategy` 사용하기

적용사례: GAN 또는 강화학습 모델링에는 Customizing이 중요.

```py
# 분산 Strategy 선언
mirrored_strategy = tf.distribute.MirroredStrategy()

# 이 scope 내에서 선언해서, Mirrored Variables 생성.
# 이 scope 내에서 optimizer 선언: 이 strategy 적용하여 모델 training한다는 의미
with mirrored_strategy.scope():
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    #model.compile(loss='mse', optimizer='sgd')
    optimizer = tf.keras.optimizers.SGD()

    # Input Pipeline 생성: strategy.scope 내에서 Dataset 정의
    dataset = (
        tf.data.Dataset.from_tensors(([1.], [1.]))
        .repeat(1000)
        .batch(global_batch_size)
    )

    # 데이터 분산 직접 선언: strategy.scope 내에서 분산 Dataset 정의
    dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)


# Training: function 정의
@tf.function
def train_step(dist_inputs):

    # training을 위한 동작 정의
    def step_fn(inputs):
        features, labels = inputs

        # Gradient 계산을 위한 record operation.
        with tf.GradientTape() as tape:
            logits = model(features)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)

            # loss 계산: `tf.nn.softmax_cross_entropy_with_logits`
            #  다만, 각 `replica`가 global_batch_size를 나눠 가졌기 때문에 
            #  그냥 저 함수를 적용하면 local_batch_size만 고려하게 되므로
            #  global_batch_size를 고려할 수 있도록 정의해야 함.
            loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)

        grads = tape.gradient(loss, model.trainable_variables)

        # 위에 선언했던 optimizer에 gradient 적용: variable 갱신(학습)을 위함.
        # 이렇게 scope 안에서 선언한 optimizer의 apply_gradient를 호출하면,
        # 각 `device`별로 gradient 계산하기 전에
        # 모든 `replica`의 gradient를 aggregation 하게 됨.
        optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

        return cross_entropy

    # 각 `device` 마다 step_fn 돌려서 loss 계산
    per_example_losses = mirrored_strategy.experimental_run_v2(
        step_fn,
        args=(dist_inputs,)
    )

    # `all-reduce` 적용하여 loss 계산
    mean_loss = mirrored_strategy.reduce(
        tf.distribute.ReduceOp.MEAN,
        per_example_losses,
        axis=0,
    )
    return mean_loss

# Training: scope 내에서 function 호출
with mirrored_strategy.scope():
    for inputs in dist_dataset:
        print(train_step(inputs))

# Optional: iterator를 이용해서 정해진 숫자만큼만 호출
train_step_num = 2
with mirrored_strategy.scope():
    iterator = iter(dist_dataset)
    for _ in range(train_step_num):
        print(train_step(next(iterator)))

# Inference(Prediction)

```

---
