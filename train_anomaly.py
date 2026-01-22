import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import get_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224*224*3, 512),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 224*224*3),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 3, 224, 224)

train_data, _ = get_dataloaders("data")
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    for images, _ in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

    print(f"Anomaly Epoch {epoch+1} completed")

torch.save(model.state_dict(), "models/autoencoder.pth")
