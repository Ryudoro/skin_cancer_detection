from model import ModelFactory
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from training import TrainingObserver, TrainingMonitor, TrainingStrategy, EvaluationStrategy
from dataset import DatasetFactory
from preprocessing import PyTorchResize, PyTorchToTensor
from torchvision import transforms

target_image_size = 150

train_transforms  = transforms.Compose([
    PyTorchResize(150),
    PyTorchToTensor()
])

val_transforms = transforms.Compose([
    PyTorchResize(150),
    PyTorchToTensor()
])

factory = DatasetFactory("HAM10000_metadata.csv", "HAM10000_images_part_1", "HAM10000_images_part_2")
train_dataset = factory.get_dataset("train",  {"train": train_transforms})
val_dataset = factory.get_dataset("validation",  {"validation": val_transforms})

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20)

img_size = 150 
num_classes = 7
model_factory = ModelFactory()
model = model_factory.get_model("SkinLesionClassifier", num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

monitor = TrainingMonitor()
observer = TrainingObserver()
monitor.register(observer)

train_strategy = TrainingStrategy()
eval_strategy = EvaluationStrategy()

train_strategy.execute(model, train_loader, criterion, optimizer, monitor, num_epochs=10)

accuracy = eval_strategy.execute(model, val_loader)