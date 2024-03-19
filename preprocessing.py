import random
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor

import random
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
import numpy as np
import tensorflow as tf

class TransformInterface:
    def __call__(self, image):
        raise NotImplementedError

class PyTorchResize(TransformInterface):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return TF.resize(image, self.size)

class PyTorchToTensor(TransformInterface):
    def __call__(self, image):
        return ToTensor()(image)

class TensorFlowTransformAdapter(TransformInterface):
    def __init__(self, tf_transform):
        self.tf_transform = tf_transform

    def __call__(self, image):
        image_np = np.array(image)
        image_tf = tf.convert_to_tensor(image_np, dtype=tf.float32)
        return self.tf_transform(image_tf)

def tf_resize(image, size):
    return tf.image.resize(image, [size, size])

def tf_to_tensor(image):
    return image / 255.0