from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        unique_labels = dataframe['dx'].unique()
        self.label_to_index = {label: index for index, label in enumerate(unique_labels)}
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']
        image = default_loader(img_path)
        label_text = self.dataframe.iloc[idx]['dx'] 
        label = self.label_to_index[label_text]
        if self.transform:
            image = self.transform(image)
            
        return image, label

class TFSkinLesionDataset:
    def __init__(self, csv_file, img_dir_1, img_dir_2, transform=None):
        self.img_dir_1 = img_dir_1
        self.img_dir_2 = img_dir_2
        self.transform = transform
        print(csv_file)
        self.metadata = csv_file
        self.metadata['path'] = self.metadata['image_id'].apply(self.find_image_path)
        self.label_to_index = {label: index for index, label in enumerate(self.metadata['dx'].unique())}

    def find_image_path(self, image_id):
        path_1 = os.path.join(self.img_dir_1, image_id + ".jpg")
        path_2 = os.path.join(self.img_dir_2, image_id + ".jpg")
        if os.path.exists(path_1):
            return path_1
        elif os.path.exists(path_2):
            return path_2
        else:
            return None

    def load_and_preprocess_image(self, path, label):

        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [150, 150])

        if self.transform:
            img = self.transform(img)

        img /= 255.0
        
        return img, label

    def get_dataset(self, dataset_type):
        train_df, val_df = train_test_split(self.metadata, test_size=0.2)
        df_selected = train_df if dataset_type == "train" else val_df

        paths = df_selected['path'].values
        labels = [self.label_to_index[label] for label in df_selected['dx'].values]
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(self.label_to_index))

        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.map(lambda path, label: tf.py_function(func=self.load_and_preprocess_image, inp=[path, label], Tout=[tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)

        return dataset.batch(32)
    
class DatasetFactory:
    def __init__(self, csv_file, img_dir_1, img_dir_2):
        self.img_dir_1 = img_dir_1
        self.img_dir_2 = img_dir_2
        self.metadata = pd.read_csv(csv_file)
        self.metadata['path'] = self.metadata['image_id'].apply(self.find_image_path)
        self.label_to_index = {label: idx for idx, label in enumerate(self.metadata['dx'].unique())}

    def find_image_path(self, image_id):
        path_1 = os.path.join(self.img_dir_1, image_id + ".jpg")
        path_2 = os.path.join(self.img_dir_2, image_id + ".jpg")
        if os.path.exists(path_1):
            return path_1
        elif os.path.exists(path_2):
            return path_2
        else:
            return None

    def load_and_preprocess_image(self, path):
        img = load_img(path, target_size=(150, 150))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0
        return img

    def get_dataset(self, dataset_type, transforms=None, framework="torch"):
        train_df, val_df = train_test_split(self.metadata, test_size=0.2)
        df = train_df if dataset_type == "train" else val_df
        
        if framework == "torch":
            dataset = SkinLesionDataset(dataframe=df, transform=transforms)
            return dataset
        elif framework == "tensorflow":
            tf_dataset = TFSkinLesionDataset(self.metadata, self.img_dir_1, self.img_dir_2, transform=transforms)
            return tf_dataset.get_dataset(dataset_type)
        else:
            raise ValueError(f"Unsupported framework: {framework}")



