import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

## Custom dataset

class PostureEmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.posture_classes = sorted(os.listdir(root_dir))
        self.emotion_classes = []

        for posture in self.posture_classes:
            posture_path = os.path.join(root_dir, posture)
            for emotion in sorted(os.listdir(posture_path)):
                if emotion not in self.emotion_classes:
                    self.emotion_classes.append(emotion)
                emotion_path = os.path.join(posture_path, emotion)
                for img in os.listdir(emotion_path):
                    if img.lower().endswith(('jpg', 'jpeg', 'png')):
                        self.samples.append({
                            'img_path': os.path.join(emotion_path, img),
                            'posture': self.posture_classes.index(posture),
                            'emotion': self.emotion_classes.index(emotion)
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(sample['posture']), torch.tensor(sample['emotion'])

## Dual-Output ShuffleNetV2 Model

class DualShuffleNet(nn.Module):
    def __init__(self, num_postures, num_emotions):
        super(DualShuffleNet, self).__init__()
        self.backbone = models.shufflenet_v2_x1_0(pretrained=True)
        self.backbone.fc = nn.Identity()

        self.shared_fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc_posture = nn.Linear(256, num_postures)
        self.fc_emotion = nn.Linear(256, num_emotions)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.stage2(x)
        x = self.backbone.stage3(x)
        x = self.backbone.stage4(x)
        x = self.backbone.conv5(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.shared_fc(x)
        return self.fc_posture(x), self.fc_emotion(x)

# Save Model & Class Info

def save_model_and_class_info(model, posture_classes, emotion_classes, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, 'shufflenet_dual_output_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to '{model_path}'")

    class_info_path = os.path.join(save_dir, 'shufflenet_class_info.pth')
    torch.save({
        'posture': posture_classes,
        'emotion': emotion_classes
    }, class_info_path)
    print(f"📄 Class info saved to '{class_info_path}'")


## Training Function

def train_model(dataset_path='dataset/', epochs=30, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = PostureEmotionDataset(dataset_path, transform=transform)
    num_posture_classes = len(dataset.posture_classes)
    num_emotion_classes = len(dataset.emotion_classes)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualShuffleNet(num_posture_classes, num_emotion_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    # Create logs directory if not exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "shufflenet_training_log.csv")

    with open(log_file_path, "w", newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["Epoch", "Train_Loss", "Val_Loss"])

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for imgs, postures, emotions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                imgs, postures, emotions = imgs.to(device), postures.to(device), emotions.to(device)
                optimizer.zero_grad()
                out_posture, out_emotion = model(imgs)
                loss = criterion(out_posture, postures) + criterion(out_emotion, emotions)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, postures, emotions in val_loader:
                    imgs, postures, emotions = imgs.to(device), postures.to(device), emotions.to(device)
                    out_posture, out_emotion = model(imgs)
                    val_loss += (criterion(out_posture, postures) + criterion(out_emotion, emotions)).item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            log_writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])
            log_file.flush()

    print(f"📄 Logs saved to {log_file_path}")

    ## Save model and class info
    save_model_and_class_info(model, dataset.posture_classes, dataset.emotion_classes)

    ## Plot loss
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Curve (ShuffleNetV2)')
    plt.legend()
    plt.show()

    return model, dataset.posture_classes, dataset.emotion_classes

if __name__ == '__main__':
    train_model('dataset/', epochs=100, batch_size=8)
