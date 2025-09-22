import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import json
import toml
import itertools
from torch.utils.data import Subset

# -----------------------
# Load Configs
# -----------------------
with open("config.json", "r") as f:
    config = json.load(f)

params = toml.load("params.toml")

with open("grid.json", "r") as f:
    grid = json.load(f)

# -----------------------
# Dataset Loader (MNIST Subset)
# -----------------------
def get_dataset(dataset_name, data_path, batch_size, subset_ratio=0.1):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_name.lower() == "mnist":
        trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    else:
        raise ValueError("Only MNIST supported in this submission")

    # Use only a subset for speed
    train_subset = Subset(trainset, range(0, int(len(trainset) * subset_ratio)))
    test_subset = Subset(testset, range(0, int(len(testset) * subset_ratio)))

    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

# -----------------------
# Model Loader (Pretrained ResNet)
# -----------------------
def get_model(arch, num_classes=10):
    if arch == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif arch == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif arch == "resnet152":
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    else:
        raise ValueError("Supported: resnet34, resnet50, resnet101, resnet152")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# -----------------------
# Training Function
# -----------------------
def train_model(model, trainloader, testloader, epochs, lr, optimizer_name, momentum):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        raise ValueError("Unsupported optimizer")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(trainloader):.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

# -----------------------
# Pipeline Integration
# -----------------------
def run_pipeline():
    dataset_name = config["data"]["dataset"]
    data_path = config["data"]["path"]
    arch = config["model"]["architecture"]

    epochs = params["training"]["epochs"]
    batch_size = params["training"]["batch_size"]

    trainloader, testloader = get_dataset(dataset_name, data_path, batch_size, subset_ratio=0.01)

    best_acc = 0
    best_config = {}

    for lr, opt, mom in itertools.product(grid["learning_rates"], grid["optimizers"], grid["momentum"]):
        print(f"\n--- Training with lr={lr}, optimizer={opt}, momentum={mom} ---")
        model = get_model(arch, num_classes=10)
        acc = train_model(model, trainloader, testloader, epochs, lr, opt, mom)

        if acc > best_acc:
            best_acc = acc
            best_config = {"lr": lr, "optimizer": opt, "momentum": mom}

    print("\n=== Best Result ===")
    print("Best Configuration:", best_config)
    print("Best Accuracy:", best_acc)

if __name__ == "__main__":
    run_pipeline()
