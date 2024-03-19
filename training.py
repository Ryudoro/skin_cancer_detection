from abc import ABC, abstractmethod
import torch

class TrainingObserver:
    def update(self, epoch, metrics):
        print(f"Epoch: {epoch}, Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.2f}%")

class TrainingMonitor:
    def __init__(self):
        self.observers = []

    def register(self, observer):
        self.observers.append(observer)

    def notify_all(self, epoch, metrics):
        for observer in self.observers:
            observer.update(epoch, metrics)


class Strategy(ABC):
    @abstractmethod
    def execute(self, model, dataloader, **kwargs):
        pass


class TrainingStrategy(Strategy):
    def execute(self, model, dataloader, criterion, optimizer, monitor, num_epochs=10):
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
    def execute(self, model, dataset, optimizer, monitor, num_epochs=10):
        loss_metric = tf.keras.metrics.Mean()
        accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
        
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
                
                if (batch + 1) % 10 == 0:
                    print(f"Batch {batch+1}, Loss: {loss_metric.result().numpy():.4f}, Accuracy: {accuracy_metric.result().numpy() * 100:.2f}%")
                    loss_metric.reset_states()
                    accuracy_metric.reset_states()
            
            monitor.notify_all(epoch, {"loss": loss_metric.result().numpy(), "accuracy": accuracy_metric.result().numpy() * 100})
            loss_metric.reset_states()
            accuracy_metric.reset_states()
            
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