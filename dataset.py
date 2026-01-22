import os
from torchvision import datasets, transforms

def get_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=transform
    )
    test_data = datasets.ImageFolder(
        os.path.join(data_dir, "test"), transform=transform
    )

    return train_data, test_data
