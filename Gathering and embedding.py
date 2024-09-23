import os
import torch
from torch.utils.data import Dataset, DataLoader # type: ignore
from PIL import Image
import torchvision.transforms as transforms

# Custom dataset class
class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.load_data()

    def load_data(self):
        for subdir in ['dev', 'train', 'test']:
            subdir_path = os.path.join(self.root_dir, subdir)
            for folder in os.listdir(subdir_path):
                folder_path = os.path.join(subdir_path, folder)
                label_path = os.path.join(folder_path, 'label.txt')
                if os.path.isdir(folder_path) and os.path.exists(label_path):
                    with open(label_path, 'r') as file:
                        label = file.read().strip()
                    for img_file in os.listdir(folder_path):
                        if img_file.endswith('.jpg') or img_file.endswith('.png'):
                            img_path = os.path.join(folder_path, img_file)
                            self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Data transformations
transform = transforms.Compose([
    transforms.Resize((210, 300)),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = SignLanguageDataset(root_dir='D:/phoenix-2014-T.v3/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Similarly, you can load dev and test datasets



import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return h[-1]

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(256 * 26 * 37, 1024)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CrossModalDiscriminator(nn.Module):
    def __init__(self, hidden_size):
        super(CrossModalDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, text_features, image_features):
        combined = torch.cat((text_features, image_features), dim=1)
        return self.fc(combined)
