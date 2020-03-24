# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Byte to Img

# %%
from keras.datasets import cifar100, fashion_mnist


# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().magic('pip install pypng')
import png

# %% [markdown]
# ## Fashion MNIST

# %%
## Fashion MNIST

"""

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

"""

from keras.datasets import fashion_mnist


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#x_train = x_train.reshape(*x_train.shape, 1)


# %%
base_dir = os.path.join(
    os.getcwd(),
    "fashion-mnist",
    "raw-images",
)


# %%
_, rows, cols = x_train.shape


filepath = os.path.join(base_dir, "train-images")
for i, img in enumerate(x_train):
    filename = os.path.join(
        filepath,
        str(i).zfill(8) + ".png",
    )
    with open(filename, "wb") as file:
        w = png.Writer(cols, rows, greyscale=True)
        w.write(file, img)


# %%
_, rows, cols = x_test.shape


filepath = os.path.join(base_dir, "test-images")
for i, img in enumerate(x_test):
    filename = os.path.join(
        filepath,
        str(i).zfill(8) + ".png",
    )
    with open(filename, "wb") as file:
        w = png.Writer(cols, rows, greyscale=True)
        w.write(file, img)


# %%
label_str = """label,desc
0,T-shirt/top
1,Trouser
2,Pullover
3,Dress
4,Coat
5,Sandal
6,Shirt
7,Sneaker
8,Bag
9,Ankle boot"""

y_dict_df = (
    pd.read_csv(StringIO(label_str), sep=",", index_col=0)
    .reset_index()
)
y_dict = y_dict_df["desc"].to_dict()


# %%
y_data_list = [y_train, y_test]
y_train_path = os.path.join(base_dir, "train-labels")
y_test_path = os.path.join(base_dir, "test-labels")
y_path_list = [y_train_path, y_test_path]

for y_data, filepath in zip(y_data_list, y_path_list):
    y_df = pd.DataFrame(y_data, columns=["label"])
    y_df.index.rename("img_num", inplace=True)
    y_df.reset_index(inplace=True)
    # y_df_described = pd.merge(y_df, y_dict_df, on="label", how="left")
    y_df["desc"] = y_df["label"].map(y_dict)
    y_df_described = y_df
    
    filename = os.path.join(
        filepath,
        os.path.basename(filepath) + ".csv",
    )
    y_df_described.to_csv(filename, index=False, header=True)

# %% [markdown]
# ## CIFAR100

# %%
from keras.datasets import cifar100


(x_train, y_train), (x_test, y_test) = cifar100.load_data(
    label_mode="fine",  # {"fine", "coarse"} == "the class", "the superclass"
    #image_data_format="channels_last",  # {"channels_first", "channels_last"}
)
(_, y_train_superclass), (_, y_test_superclass) = cifar100.load_data(
    label_mode="coarse",
)


# %%
fine_labels = [
'apple', # id 0
'aquarium_fish',
'baby',
'bear',
'beaver',
'bed',
'bee',
'beetle',
'bicycle',
'bottle',
'bowl',
'boy',
'bridge',
'bus',
'butterfly',
'camel',
'can',
'castle',
'caterpillar',
'cattle',
'chair',
'chimpanzee',
'clock',
'cloud',
'cockroach',
'couch',
'crab',
'crocodile',
'cup',
'dinosaur',
'dolphin',
'elephant',
'flatfish',
'forest',
'fox',
'girl',
'hamster',
'house',
'kangaroo',
'computer_keyboard',
'lamp',
'lawn_mower',
'leopard',
'lion',
'lizard',
'lobster',
'man',
'maple_tree',
'motorcycle',
'mountain',
'mouse',
'mushroom',
'oak_tree',
'orange',
'orchid',
'otter',
'palm_tree',
'pear',
'pickup_truck',
'pine_tree',
'plain',
'plate',
'poppy',
'porcupine',
'possum',
'rabbit',
'raccoon',
'ray',
'road',
'rocket',
'rose',
'sea',
'seal',
'shark',
'shrew',
'skunk',
'skyscraper',
'snail',
'snake',
'spider',
'squirrel',
'streetcar',
'sunflower',
'sweet_pepper',
'table',
'tank',
'telephone',
'television',
'tiger',
'tractor',
'train',
'trout',
'tulip',
'turtle',
'wardrobe',
'whale',
'willow_tree',
'wolf',
'woman',
'worm',
]

coarse_labels = {
'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
'household electrical device': ['clock', 'computer_keyboard', 'lamp', 'telephone', 'television'],
'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
'people': ['baby', 'boy', 'girl', 'man', 'woman'],
'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}


# %%
fine_labels_dict = {i: label for i, label in enumerate(fine_labels)}

superclass_dict = {
    value: key for key, value_list in coarse_labels.items()
     for value in value_list
}
superclass_int_dict = {key: i for i, key in enumerate(sorted(coarse_labels))}


# %%
y_df = pd.DataFrame(
    {
        "label": y_train.flatten(),
        "super_label": y_train_superclass.flatten(),
    }
)


# %%
y_train_df = pd.DataFrame(y_train[:10], columns=["labels"])
y_train_df["labels_description"] = y_train_df["labels"].map(fine_labels_dict)
y_train_df["super_labels"] = y_train_df["labels_description"].map(superclass_dict)
y_train_df["super_labels_int"] = y_train_df["super_labels"].map(superclass_int_dict)


# %%
base_dir = os.path.join(
    os.getcwd(),
    "cifar100",
    "raw-images",
)


# %%
_, rows, cols, channels = x_train.shape


filepath = os.path.join(base_dir, "train-images")
os.makedirs(filepath, exist_ok=True)
for i, img in enumerate(x_train):
    filename = os.path.join(
        filepath,
        str(i).zfill(8) + ".png",
    )
 
    plt.imsave(filename, img)


# %%
_, rows, cols, channels = x_test.shape


filepath = os.path.join(base_dir, "test-images")
os.makedirs(filepath, exist_ok=True)
for i, img in enumerate(x_test):
    filename = os.path.join(
        filepath,
        str(i).zfill(8) + ".png",
    )
    plt.imsave(filename, img)


# %%
y_data_list = [[y_train, y_train_superclass], [y_test, y_test_superclass]]
y_train_path = os.path.join(base_dir, "train-labels")
y_test_path = os.path.join(base_dir, "test-labels")
y_path_list = [y_train_path, y_test_path]

for y_data, filepath in zip(y_data_list, y_path_list):
    y_df = pd.DataFrame(y_train, columns=["label"])
    y_data_label, y_data_superclass = y_data
    y_df = pd.DataFrame(
        {
            "label": y_data_label.flatten(),
            "super_label": y_data_superclass.flatten(),
        }
    )
    y_df.index.rename("img_num", inplace=True)
    y_df.reset_index(inplace=True)
    # y_df_described = pd.merge(y_df, y_dict_df, on="label", how="left")
    y_df["label_desc"] = y_df["label"].map(fine_labels_dict)
    y_df["super_label_desc"] = y_df["super_label"].map(superclass_int_dict)
    #y_train_df["super_label_desc"] = y_train_df["label_desc"].map(superclass_dict)
    #y_train_df["super_label_int"] = y_train_df["super_label_desc"].map(superclass_int_dict)

    y_df_described = y_df

    os.makedirs(filepath, exist_ok=True)
    filename = os.path.join(
        filepath,
        os.path.basename(filepath) + ".csv",
    )
    y_df_described.to_csv(filename, index=False, header=True)


# %%

