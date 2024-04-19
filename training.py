from abc import ABC, abstractmethod
import torch
import csv
import datetime
import os

class TrainingObserver:
    def update(self, epoch, metrics):
        print(f"Epoch: {epoch}, Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.2f}%")



class TrainingMonitor:
    def __init__(self, model_dir='models', log_dir='logs'):
        self.observers = []
        self.model_dir = model_dir
        self.log_dir = log_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        # self.log_file_path = os.path.join(self.log_dir, f"training_metrics_{self._current_timestamp()}.csv")
        self.fieldnames = ['epoch', 'loss', 'accuracy', 'precision', 'recall', 'f1_score']
        self.log_file_path = self._create_log_file()
        with open(self.log_file_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()

    def _create_log_file(self):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = os.path.join(self.log_dir, f"training_metrics_{timestamp}.csv")
        with open(log_file_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
        return log_file_path
    
    def register(self, observer):
        self.observers.append(observer)

    def notify_all(self, epoch, metrics):
        for observer in self.observers:
            observer.update(epoch, metrics)
        self._log_metrics(epoch, metrics)
        if metrics.get('save'):
            self._save_model(metrics['model'])

    def _log_metrics(self, epoch, metrics):

        csv_metrics = {key: metrics[key] for key in self.fieldnames if key in metrics}
        csv_metrics['epoch'] = epoch
        with open(self.log_file_path, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(csv_metrics)

    def _save_model(self, model):
        model_path = os.path.join(self.model_dir, f"model_{self._current_timestamp()}")
        model.save(model_path)

    def _current_timestamp(self):
        return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


class Strategy(ABC):
    @abstractmethod
    def execute(self, model, dataloader, **kwargs):
        pass


class TrainingStrategy(Strategy):
    def execute(self, model, dataloader, criterion, optimizer, monitor, num_epochs=10, save  = False):
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch+1}/{num_epochs}")
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                print(f"Batch processed.")
            
            epoch_loss = running_loss / len(dataloader)
            epoch_accuracy = (correct_predictions / total_predictions) * 100
            monitor.notify_all(epoch, {"loss": epoch_loss, "accuracy": epoch_accuracy})
        if save:
            model.save("model_result")
class EvaluationStrategy(Strategy):
    def execute(self, model, dataloader, **kwargs):
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy}%')
        return accuracy
    

import tensorflow as tf

class TFTrainingStrategy(Strategy):
    def execute(self, model, dataset, optimizer, monitor, num_epochs=10, save  = False):
        loss_metric = tf.keras.metrics.Mean()
        accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
        precision_metric = tf.keras.metrics.Precision()
        recall_metric = tf.keras.metrics.Recall()
        f1_score_metric = tf.keras.metrics.Mean()

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            for batch, (inputs, labels) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    predictions = model(inputs, training=True)
                    loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                loss_metric.update_state(loss)
                accuracy_metric.update_state(labels, predictions)
                precision_metric.update_state(labels, predictions)
                recall_metric.update_state(labels, predictions)
                # Calculate F1 Score
                precision = precision_metric.result()
                recall = recall_metric.result()
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
                f1_score_metric.update_state(f1_score)

                if (batch + 1) % 10 == 0:
                    print(f"Batch {batch+1}, Loss: {loss_metric.result().numpy():.4f}, Accuracy: {accuracy_metric.result().numpy() * 100:.2f}%")
                    loss_metric.reset_states()
                    accuracy_metric.reset_states()
            
            metrics = {
                "loss": loss_metric.result().numpy(),
                "accuracy": accuracy_metric.result().numpy() * 100,
                "precision": precision_metric.result().numpy(),
                "recall": recall_metric.result().numpy(),
                "f1_score": f1_score_metric.result().numpy(),
                "model": model if save else None,
                "save": save
            }
            monitor.notify_all(epoch, metrics)
            loss_metric.reset_states()
            accuracy_metric.reset_states()
            precision_metric.reset_states()
            recall_metric.reset_states()
            f1_score_metric.reset_states()
        if save:
            model.save("model_result_tf")
            

class TFDualTrainingStrategy:
    def execute(self, model, dataset, optimizer, monitor, num_epochs=10, save=False):
        loss_metric = tf.keras.metrics.Mean()
        accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Adaptation pour déballer correctement les valeurs
            for batch, (images, metadata, labels) in enumerate(dataset):
                inputs = {'image': images, 'metadata': metadata}  # Préparer les données comme attendu par le modèle
                
                with tf.GradientTape() as tape:
                    predictions = model(inputs, training=True)
                    loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
                
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                loss_metric.update_state(loss)
                accuracy_metric.update_state(labels, predictions)
                
                if (batch + 1) % 10 == 0:
                    print(f"Batch {batch+1}, Loss: {loss_metric.result().numpy():.4f}, Accuracy: {accuracy_metric.result().numpy() * 100:.2f}%")
                    loss_metric.reset_states()
                    accuracy_metric.reset_states()
            
            monitor.notify_all(epoch, {"loss": loss_metric.result().numpy(), "accuracy": accuracy_metric.result().numpy() * 100})
            loss_metric.reset_states()
            accuracy_metric.reset_states()
        
        if save:
            model.save("model_result_tf_dual")

class TFEvaluationStrategy(Strategy):
    def execute(self, model, dataset, **kwargs):
        total = 0
        correct = 0
        for inputs, labels in dataset:
            predictions = model(inputs, training=False)
            predicted_labels = tf.argmax(predictions, axis=1)
            true_labels = tf.argmax(labels, axis=1)
            correct += tf.reduce_sum(tf.cast(predicted_labels == true_labels, tf.float32))
            total += inputs.shape[0]
        accuracy = correct / total
        print(f'Accuracy: {accuracy.numpy() * 100}%')
        return accuracy.numpy() * 100