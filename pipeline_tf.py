import tensorflow as tf
from dataset import DatasetFactory
from model import ModelFactory
from training import TrainingMonitor, TFTrainingStrategy, TFDualTrainingStrategy
import numpy as np

factory = DatasetFactory(csv_file='all_paths_metadata_df.csv', img_dir_1='dataset_HAM10000/HAM10000_images_part_1', img_dir_2='dataset_HAM10000/HAM10000_images_part_2')
pytorch_dataset = factory.get_dataset('train', None, framework='torch')

tf_dataset = factory.get_dataset('train', None, framework='tensorflow', dataset_class="normal_input")

tf_model = ModelFactory().get_model('PretrainedEfficientSkinLesionClassifier', num_classes=7, framework='tensorflow')

optimizer = tf.keras.optimizers.Adam()

tf_monitor = TrainingMonitor()
# tf_training_strategy = TFDualTrainingStrategy()
tf_training_strategy = TFTrainingStrategy()
tf_training_strategy.execute(tf_model, tf_dataset, optimizer, tf_monitor, num_epochs=10, save = True)


# for images, metadata, labels in tf_dataset.take(1):
#     print("Images batch shape:", images.numpy().shape)
#     print("Metadata batch shape:", metadata.numpy().shape)
#     print("Labels batch shape:", labels.numpy().shape)
#     print("Images dtype:", images.numpy().dtype)
#     print("Metadata dtype:", metadata.numpy().dtype)
#     print("Labels dtype:", labels.numpy().dtype)


# import matplotlib.pyplot as plt

# def plot_images(images, metadata, labels):
#     plt.figure(figsize=(10, 10))
#     for i in range(9):
#         plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i])
#         plt.title(f'Label: {np.argmax(labels[i])}')
#         plt.axis('off')
#     plt.show()

# # Prendre un batch de données et afficher des images
# for images, metadata, labels in tf_dataset.take(1):
#     plot_images(images.numpy(), metadata.numpy(), labels.numpy())


# for images, metadata, labels in tf_dataset.take(1):
#     print("Sample metadata:", metadata.numpy()[:5])  # Afficher les métadonnées de 5 échantillons
#     print("Sample labels:", labels.numpy()[:5])