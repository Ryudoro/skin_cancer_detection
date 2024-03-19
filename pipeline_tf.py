import tensorflow as tf
from dataset import DatasetFactory
from model import ModelFactory
from training import TrainingMonitor, TFTrainingStrategy

factory = DatasetFactory(csv_file='HAM10000_metadata.csv', img_dir_1='HAM10000_images_part_1', img_dir_2='HAM10000_images_part_2')
pytorch_dataset = factory.get_dataset('train', None, framework='torch')

tf_dataset = factory.get_dataset('train', None, framework='tensorflow')

tf_model = ModelFactory().get_model('PretrainedSkinLesionClassifier', num_classes=7, framework='tensorflow')

optimizer = tf.keras.optimizers.Adam()

tf_monitor = TrainingMonitor()
tf_training_strategy = TFTrainingStrategy()
tf_training_strategy.execute(tf_model, tf_dataset, optimizer, tf_monitor, num_epochs=10)