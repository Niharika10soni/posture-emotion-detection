import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import psutil
import cv2
import csv
import numpy as np
from torchvision.models import shufflenet_v2_x1_0
from PIL import Image
from threading import Thread
from sklearn.metrics import classification_report, confusion_matrix


class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.thread = Thread(target=self.update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()

## Model

class DualShuffleNet(nn.Module):
    def __init__(self, num_postures, num_emotions):
        super(DualShuffleNet, self).__init__()
        self.backbone = shufflenet_v2_x1_0(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.shared_fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc_posture = nn.Linear(256, num_postures)
        self.fc_emotion = nn.Linear(256, num_emotions)

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared_fc(x)
        return self.fc_posture(x), self.fc_emotion(x)

## Model loading 

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_info = torch.load("models/shufflenet_class_info.pth")
    posture_classes = class_info["posture"]
    emotion_classes = class_info["emotion"]

    model = DualShuffleNet(len(posture_classes), len(emotion_classes))
    model.load_state_dict(torch.load("models/shufflenet_dual_output_model.pt", map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    return model, transform, posture_classes, emotion_classes, device

## Webcam Inference 

def run_inference_webcam():
    model, transform, posture_classes, emotion_classes, device = load_model()

    os.makedirs("logs", exist_ok=True)
    log_path = "logs/realtime_log_shufflenet.csv"
    log_file = open(log_path, "w", newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(["Timestamp", "FPS", "CPU (%)", "RAM (%)", "Posture", "Emotion"])

    vs = VideoStream().start()
    cv2.namedWindow("Posture & Emotion Detection", cv2.WINDOW_NORMAL)
    prev_time = time.time()

    try:
        with torch.no_grad():
            while True:
                ret, frame = vs.read()
                if not ret or frame is None:
                    break

                h, w, _ = frame.shape
                min_dim = min(h, w)
                frame_cropped = frame[(h - min_dim) // 2:(h + min_dim) // 2,
                                      (w - min_dim) // 2:(w + min_dim) // 2]
                img_pil = Image.fromarray(cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB))
                img_tensor = transform(img_pil).unsqueeze(0).to(device)

                out_posture, out_emotion = model(img_tensor)
                pred_posture = posture_classes[torch.argmax(out_posture).item()]
                pred_emotion = emotion_classes[torch.argmax(out_emotion).item()]

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time + 1e-6)
                prev_time = curr_time

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory().percent

                overlay = [
                    f'Time: {timestamp}',
                    f'FPS: {fps:.2f}',
                    f'CPU: {cpu:.1f}%',
                    f'RAM: {ram:.1f}%',
                    f'Posture: {pred_posture}',
                    f'Emotion: {pred_emotion}'
                ]
                for i, line in enumerate(overlay):
                    cv2.putText(frame, line, (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Posture & Emotion Detection", frame)

                log_writer.writerow([timestamp, f"{fps:.2f}", f"{cpu:.1f}", f"{ram:.1f}", pred_posture, pred_emotion])
                log_file.flush()

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    finally:
        vs.stop()
        log_file.close()
        cv2.destroyAllWindows()
        print(f"✅ Inference stopped. Log saved to: {log_path}")

## Batch Inference

def run_batch_inference(dataset_path="mini_dataset", output_dir="output_batch"):
    model, transform, posture_classes, emotion_classes, device = load_model()

    os.makedirs("logs", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    log_path = "logs/batch_inference_log.csv"
    csv_file = open(log_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["Image", "True_Posture", "Pred_Posture", "True_Emotion", "Pred_Emotion", "Correct"])

    y_true_posture, y_pred_posture = [], []
    y_true_emotion, y_pred_emotion = [], []
    correct = 0
    total = 0

    with torch.no_grad():
        for posture in os.listdir(dataset_path):
            posture_path = os.path.join(dataset_path, posture)
            if not os.path.isdir(posture_path): continue
            for emotion in os.listdir(posture_path):
                emotion_path = os.path.join(posture_path, emotion)
                if not os.path.isdir(emotion_path): continue
                for img_name in os.listdir(emotion_path):
                    img_path = os.path.join(emotion_path, img_name)
                    try:
                        img = Image.open(img_path).convert("RGB")
                    except:
                        print(f"⚠️ Skipping unreadable image: {img_path}")
                        continue

                    img_tensor = transform(img).unsqueeze(0).to(device)
                    out_posture, out_emotion = model(img_tensor)
                    pred_posture = posture_classes[torch.argmax(out_posture).item()]
                    pred_emotion = emotion_classes[torch.argmax(out_emotion).item()]

                    is_correct = (pred_posture == posture and pred_emotion == emotion)
                    correct += int(is_correct)
                    total += 1

                    writer.writerow([
                        img_name, posture, pred_posture,
                        emotion, pred_emotion, is_correct
                    ])

                    y_true_posture.append(posture)
                    y_pred_posture.append(pred_posture)
                    y_true_emotion.append(emotion)
                    y_pred_emotion.append(pred_emotion)

                    # Annotate image
                    annotated_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    cv2.putText(annotated_img, f"True: {posture}/{emotion}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(annotated_img, f"Pred: {pred_posture}/{pred_emotion}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 0) if is_correct else (0, 0, 255), 2)

                    # Save annotated image
                    save_name = f"{os.path.splitext(img_name)[0]}__true_{posture}_{emotion}__pred_{pred_posture}_{pred_emotion}.jpg"
                    save_path = os.path.join(output_dir, save_name)
                    cv2.imwrite(save_path, annotated_img)

    csv_file.close()
    accuracy = correct / total if total else 0
    print(f"\n✅ Batch inference complete. Accuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"📂 Annotated images saved to: {output_dir}")

    print("\n📊 Posture Classification Report:")
    print(classification_report(y_true_posture, y_pred_posture, labels=posture_classes))
    print("\n📊 Emotion Classification Report:")
    print(classification_report(y_true_emotion, y_pred_emotion, labels=emotion_classes))
    print("\n📉 Posture Confusion Matrix:")
    print(confusion_matrix(y_true_posture, y_pred_posture, labels=posture_classes))
    print("\n📉 Emotion Confusion Matrix:")
    print(confusion_matrix(y_true_emotion, y_pred_emotion, labels=emotion_classes))
    print(f"\n📄 Logged to: {log_path}")
