"""
face_identification_colab.py
----------------------------
Robust face identification system for Colab.
Requirements:
    pip install deepface==0.0.95 gspread opencv-python-headless numpy
"""

import os
import time
import pickle
import numpy as np
import cv2
from base64 import b64decode

# Colab-specific imports
#from IPython.display import display, Javascript
#from google.colab import files
#from google.colab.output import eval_js

# Google Sheets
import gspread
from google.auth.exceptions import GoogleAuthError
from deepface import DeepFace

import matplotlib.pyplot as plt
import cv2


# ---------------- CONFIG ----------------
SPREADSHEET_NAME = "facerecognitionapp"
WORKSHEET_NAME = "Attendance"
SERVICE_ACCOUNT_FILE = "face_recognition.json"
ENCODING_FILE_NAME = "face_embeddings.pkl"
DETECTOR_BACKEND = "retinaface"
MODEL_NAME = "Facenet"
TOLERANCE = 0.25  # strict threshold for robust recognition
MIN_SAMPLES_PER_PERSON = 5
# ---------------------------------------

# --- Global variables ---
known_face_embeddings_dict = {}
worksheet = None

# ----------------- UTILITY FUNCTIONS -----------------

def init_gspread():
    """Initialize Google Sheets connection."""
    global worksheet
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        print("Upload your service account JSON file.")
        uploaded = files.upload()
        if SERVICE_ACCOUNT_FILE not in uploaded:
            raise FileNotFoundError(f"{SERVICE_ACCOUNT_FILE} not uploaded.")
    try:
        gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
        spreadsheet = gc.open(SPREADSHEET_NAME)
        worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
        print(f"✅ Connected to Google Sheet '{SPREADSHEET_NAME}' / '{WORKSHEET_NAME}'")
    except gspread.SpreadsheetNotFound:
        raise RuntimeError(f"Spreadsheet '{SPREADSHEET_NAME}' not found.")
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=WORKSHEET_NAME, rows="1000", cols="3")
        worksheet.append_row(["Timestamp", "Person Name", "Log Type"])
        print(f"✅ Worksheet '{WORKSHEET_NAME}' created and initialized.")

def log_attendance(name, log_type="Attendance"):
    """Log attendance or enrollment into Google Sheet."""
    if worksheet:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        worksheet.insert_row([timestamp, name, log_type], 2)
        print(f"🚀 Logged: {name} ({log_type}) at {timestamp}")
    else:
        print("⚠️ Worksheet not connected. Skipping logging.")

def load_encodings():
    """Load embeddings from file."""
    global known_face_embeddings_dict
    if os.path.exists(ENCODING_FILE_NAME):
        with open(ENCODING_FILE_NAME, 'rb') as f:
            known_face_embeddings_dict = pickle.load(f)
        print(f"✅ Loaded {len(known_face_embeddings_dict)} known faces.")
    else:
        known_face_embeddings_dict = {}
        print("✅ No existing embeddings. Starting fresh.")

def save_encodings():
    """Save embeddings to file."""
    with open(ENCODING_FILE_NAME, 'wb') as f:
        pickle.dump(known_face_embeddings_dict, f)
    print(f"✅ Saved embeddings for {len(known_face_embeddings_dict)} people.")

# ----------------- FACE FUNCTIONS -----------------
def show_frame(frame, title="Face Detection"):
    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,6))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

import cv2

def take_photo_frame():
    """Capture a frame from the webcam (works locally, not Colab JS)."""
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return None

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Failed to capture frame")
        return None

    return frame

def detect_and_embed_faces(frame):
    """Detect faces and generate embeddings using DeepFace."""
    try:
        results = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True
        )
    except Exception as e:
        print(f"DeepFace detection error: {e}")
        return [], []

    face_locations, embeddings = [], []
    for res in results:
        if 'face' in res and res['face'] is not None:
            x, y, w, h = res['facial_area']['x'], res['facial_area']['y'], res['facial_area']['w'], res['facial_area']['h']
            top, right, bottom, left = y, x + w, y + h, x
            face_locations.append((top, right, bottom, left))
            emb = DeepFace.represent(img_path=res['face'], model_name=MODEL_NAME, enforce_detection=False)[0]['embedding']
            embeddings.append(emb)
    return face_locations, embeddings

def get_average_embeddings():
    avg_embeddings = {}
    for name, vectors in known_face_embeddings_dict.items():
        if len(vectors) > 0:
            avg_embeddings[name] = np.mean(np.array(vectors), axis=0)
    return avg_embeddings

def recognize_face_robust(embedding):
    best_match = "Unknown"
    best_distance = float('inf')

    # Compare embedding to ALL saved embeddings for each person
    for name, emb_list in known_face_embeddings_dict.items():
        distances = [
            np.linalg.norm(np.array(emb) - np.array(embedding)) 
            for emb in emb_list
        ]

        if not distances:
            continue

        person_min = min(distances)

        if person_min < best_distance:
            best_distance = person_min
            best_match = name

    # --- 🔥 IMPORTANT NEW CHECK (place it here!) ---
    # Require minimum 3 samples for a person before recognizing them
    if len(known_face_embeddings_dict.get(best_match, [])) < 3:
        return "Unknown", best_distance
    # -------------------------------------------------

    # Final distance threshold check
    if best_distance > TOLERANCE:
        return "Unknown", best_distance

    return best_match, best_distance


# ----------------- MAIN CAPTURE LOOP -----------------
def capture_and_process():
    capture = input("Press 'Y' to capture a photo (or anything else to cancel): ").strip().upper()
    if capture != 'Y':
        print("Capture cancelled.")
        return

    print("Capturing... (webcam will open in browser/Colab)")
    frame = take_photo_frame()
    if frame is None:
        print("ERROR: Could not capture image. Try again.")
        return

    face_locations, embeddings = detect_and_embed_faces(frame)
    if not embeddings:
        print("No faces detected. Try again with clearer image.")
        return

    for loc, emb in zip(face_locations, embeddings):
        name, distance = recognize_face_robust(emb)
        top, right, bottom, left = loc
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left + 5, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if name != "Unknown":
            log_attendance(name, "Attendance")
        else:
            new_name = input(f"Unknown face detected (distance: {distance:.3f}). Enter name (or skip): ").strip()
            if new_name:
                if new_name not in known_face_embeddings_dict:
                    known_face_embeddings_dict[new_name] = []
                    log_attendance(new_name, "Enrollment")
                known_face_embeddings_dict[new_name].append(emb)
                save_encodings()
                remaining = max(0, MIN_SAMPLES_PER_PERSON - len(known_face_embeddings_dict[new_name]))
                if remaining > 0:
                    print(f"⚠️ Add {remaining} more samples for '{new_name}' to improve recognition.")

    # Show image with bounding boxes
    show_frame(frame)


# ----------------- ENTRY POINT -----------------
if __name__ == "__main__":
    load_encodings()
    init_gspread()
    print("Ready. Press Ctrl+C to exit.")
    while True:
        capture_and_process()
