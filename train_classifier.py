import torch
from torch import nn, optim
from torchvision import models
from torch.utils.data import DataLoader
from dataset import get_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"

train_data, test_data = get_dataloaders("data")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed")

torch.save(model.state_dict(), "models/classifier.pth")
