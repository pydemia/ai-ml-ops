
import os
import tensorflow as tf

"""
https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet/resnet_model.py
"""
# %% Format: CKPT
MODEL_CKPT_DIR = "models/aiplatform-jobs/builtin-image-classification/test-000"
MODEL_DIR = f"{MODEL_CKPT_DIR}/model"

tf.train.latest_checkpoint(MODEL_CKPT_DIR)

resnet50 = tf.keras.applications.ResNet50V2()


os.makedirs(MODEL_DIR, exist_ok=True)
tf.saved_model.save(resnet50, f"{MODEL_CKPT_DIR}/model")

print("finished")
# %% Format: Saved Model

loaded_model = tf.keras.models.load_model(MODEL_DIR)
