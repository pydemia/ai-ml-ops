# Data

## Fashion MNIST

```py

from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

```

| Label | Description |
| :---: | :---------: |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## CIFAR 100

```py

from keras.datasets import cifar100


(x_train, y_train), (x_test, y_test) = cifar100.load_data(
    label_mode="fine",
    image_data_format="channels_last",  # {"channels_first", "channels_last"}
)

```