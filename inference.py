import torch
from torchvision import transforms, models
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/classifier.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    output = model(image)
    return torch.argmax(output, dim=1).item()

print("Prediction:", predict("data/test/normal/sample.jpg"))
