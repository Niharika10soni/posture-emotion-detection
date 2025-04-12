import cv2
import os
from datetime import datetime

# Define posture and emotion categories
posture_options = {
    'u': "upright",
    'h': "hunched"
}

emotion_options = {
    'r': "relaxed",
    's': "stressed",
    'a': "angry"
}

# Set base dir
base_path = "dataset"

# Create all required folders like dataset/upright/relaxed etc.
for posture in posture_options.values():
    for emotion in emotion_options.values():
        os.makedirs(os.path.join(base_path, posture, emotion), exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access the camera")
    exit()

print("✅ Camera initialized.")
print("➡️  First, select a posture:")
print("   'u' = Upright")
print("   'h' = Hunched")
print("   'q' = Quit")

current_posture = None

def generate_filename(folder):
    """Generate a unique filename using timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(folder, f"{timestamp}.jpg")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame from camera")
        break

    # Display current instructions
    instructions = f"Posture: {current_posture if current_posture else 'Not selected'} | 'u'/'h' posture | 'r'/'s'/'a' emotion | 'q' quit"
    frame_display = frame.copy()
    cv2.putText(frame_display, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("📸 Dataset Collector", frame_display)
    key = cv2.waitKey(10) & 0xFF

    if key == ord('q'):
        print("👋 Quitting...")
        break

    key_char = chr(key)

    # Select posture
    if key_char in posture_options:
        current_posture = posture_options[key_char]
        print(f"✅ Posture set to: {current_posture}")
        print("➡️  Now press:")
        print("   'r' = Relaxed")
        print("   's' = Stressed")
        print("   'a' = Angry")

    # Save frame based on emotion if posture is selected
    elif key_char in emotion_options and current_posture:
        emotion = emotion_options[key_char]
        folder = os.path.join(base_path, current_posture, emotion)
        filename = generate_filename(folder)
        success = cv2.imwrite(filename, frame)
        if success:
            print(f"✅ Saved: {filename}")
        else:
            print(f"❌ Failed to save: {filename}")

    elif key_char in emotion_options and not current_posture:
        print("⚠️ Please select a posture first using 'u' or 'h'.")

cap.release()
cv2.destroyAllWindows()

