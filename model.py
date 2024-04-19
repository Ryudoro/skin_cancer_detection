import torch.nn as nn
import torch.nn.functional as F
import torch
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2,EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Concatenate, Input
import tensorflow_hub as hub

def conv_output_size(input_size, kernel_size, padding, stride):
    return (input_size - kernel_size + 2*padding) // stride + 1



class SkinLesionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SkinLesionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5)) 
        
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x) 
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


class TFSkinLesionClassifier(tf.keras.Model):
    def __init__(self, num_classes):
        super(TFSkinLesionClassifier, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 3))
        self.pool = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.adaptive_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.fc2 = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class PretrainedSkinLesionClassifier(tf.keras.Model):
    def __init__(self, num_classes):
        super(PretrainedSkinLesionClassifier, self).__init__()
        self.base_model = MobileNetV2(input_shape=(150, 150, 3),
                                      include_top=False,
                                      weights='imagenet')
        self.global_average_pooling = GlobalAveragePooling2D()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(num_classes, activation='softmax')
        
        self.base_model.trainable = False

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.global_average_pooling(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class PretrainedEfficientSkinLesionClassifier(tf.keras.Model):
    def __init__(self, num_classes):
        super(PretrainedEfficientSkinLesionClassifier, self).__init__()
        self.base_model = EfficientNetB0(input_shape=(150, 150, 3),
                                         include_top=False,
                                         weights='imagenet')
  
        self.global_average_pooling = GlobalAveragePooling2D()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(num_classes, activation='softmax')

        self.base_model.trainable = False

    def call(self, inputs):
        x = self.base_model(inputs, training=False) 
        x = self.global_average_pooling(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class DualInputMobileNetSkinLesionClassifier(tf.keras.Model):
    def __init__(self, num_classes, num_metadata_features):
        super(DualInputMobileNetSkinLesionClassifier, self).__init__()
        self.base_model = MobileNetV2(input_shape=(150, 150, 3),
                                      include_top=False,
                                      weights='imagenet')
        self.global_average_pooling = GlobalAveragePooling2D()
        
        # Layers for processing metadata
        self.meta_input = Input(shape=(num_metadata_features,))
        self.meta_dense1 = Dense(32, activation='relu')

        # Layers for final prediction
        self.concat = Concatenate()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        images = inputs['image']
        metadata = inputs['metadata']

        x1 = self.base_model(images, training =False)
        x1 = self.global_average_pooling(x1)
        x2 = self.meta_dense1(metadata)
        
        x = self.concat([x1, x2])
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class DualInputEfficientNetSkinLesionClassifier(tf.keras.Model):
    def __init__(self, num_classes, num_metadata_features):
        super(DualInputEfficientNetSkinLesionClassifier, self).__init__()
        self.efficient_net = EfficientNetB0(include_top=False, input_shape=(150, 150, 3), weights='imagenet', pooling='avg')
        self.efficient_net.trainable = False  # Freeze the base model
        
        # Layers for processing metadata
        self.meta_input = Input(shape=(num_metadata_features,))
        self.meta_dense1 = Dense(32, activation='relu')

        # Layers for final prediction
        self.concat = Concatenate()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        images = inputs['image']
        metadata = inputs['metadata']

        x1 = self.efficient_net(images)
        x2 = self.meta_dense1(metadata)
        
        x = self.concat([x1, x2])
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class PretrainedVisionTransformerSkinLesionClassifier(tf.keras.Model):
    def __init__(self, num_classes):
        super(PretrainedVisionTransformerSkinLesionClassifier, self).__init__()
        self.vit_model = hub.KerasLayer("https://tfhub.dev/google/vit-base/patch16-224-in21k/1",
                                        trainable=False)
        
        self.global_average_pooling = GlobalAveragePooling2D()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.vit_model(inputs)
        x = self.global_average_pooling(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class ModelFactory:
    def get_model(self, model_name, num_classes, framework='torch', num_metadata_features=3):
        if framework == 'torch':
            if model_name == "SkinLesionClassifier":
                return SkinLesionClassifier(num_classes)
        elif framework == 'tensorflow':
            if model_name == "SkinLesionClassifier":
                return TFSkinLesionClassifier(num_classes)
            elif model_name == "PretrainedSkinLesionClassifier":
                return PretrainedSkinLesionClassifier(num_classes)
            elif model_name == "PretrainedEfficientSkinLesionClassifier":
                    return PretrainedEfficientSkinLesionClassifier(num_classes)
            elif model_name == "PretrainedVisionTransformerSkinLesionClassifier":
                return PretrainedVisionTransformerSkinLesionClassifier(num_classes)
            elif model_name == "DualInputEfficientNetSkinLesionClassifier":
                return DualInputEfficientNetSkinLesionClassifier(num_classes, num_metadata_features)
            
            elif model_name == "DualInputMobileNetSkinLesionClassifier":
                return DualInputMobileNetSkinLesionClassifier(num_classes, num_metadata_features)
        else:
            raise ValueError("Unsupported framework: {}".format(framework))
        
