"""
Face Recognition System with DeepFace and Google Sheets
Using Gradio for web interface
"""

import gradio as gr
import cv2
import numpy as np
import pickle
import os
import time
import gspread
from deepface import DeepFace

# ---------------- CONFIG ----------------
SPREADSHEET_NAME = "facerecognitionapp"
WORKSHEET_NAME = "Attendance"
SERVICE_ACCOUNT_FILE = "face_recognition.json"
ENCODING_FILE_NAME = "face_embeddings.pkl"
# DETECTOR_BACKEND = "retinaface"
DETECTOR_BACKEND = "opencv"
MODEL_NAME = "Facenet"
TOLERANCE = 0.25
MIN_SAMPLES_PER_PERSON = 3
# ---------------------------------------

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_embeddings_dict = {}
        self.worksheet = None
        self.load_encodings()
        self.init_gspread()
    
    def init_gspread(self):
        """Initialize Google Sheets connection."""
        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            print(f"⚠️ Warning: {SERVICE_ACCOUNT_FILE} not found. Google Sheets logging disabled.")
            return
        
        try:
            gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
            spreadsheet = gc.open(SPREADSHEET_NAME)
            self.worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
            print(f"✅ Connected to Google Sheet '{SPREADSHEET_NAME}' / '{WORKSHEET_NAME}'")
        except gspread.SpreadsheetNotFound:
            print(f"⚠️ Spreadsheet '{SPREADSHEET_NAME}' not found. Create it first.")
        except gspread.WorksheetNotFound:
            spreadsheet = gc.open(SPREADSHEET_NAME)
            self.worksheet = spreadsheet.add_worksheet(title=WORKSHEET_NAME, rows="1000", cols="3")
            self.worksheet.append_row(["Timestamp", "Person Name", "Log Type"])
            print(f"✅ Worksheet '{WORKSHEET_NAME}' created.")
        except Exception as e:
            print(f"⚠️ Google Sheets connection error: {e}")
    
    def log_attendance(self, name, log_type="Attendance"):
        """Log attendance or enrollment into Google Sheet."""
        if self.worksheet:
            try:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                self.worksheet.insert_row([timestamp, name, log_type], 2)
                return f"✅ Logged: {name} ({log_type}) at {timestamp}"
            except Exception as e:
                return f"⚠️ Logging failed: {e}"
        return "⚠️ Google Sheets not connected"
    
    def load_encodings(self):
        """Load embeddings from file."""
        if os.path.exists(ENCODING_FILE_NAME):
            with open(ENCODING_FILE_NAME, 'rb') as f:
                self.known_face_embeddings_dict = pickle.load(f)
            print(f"✅ Loaded {len(self.known_face_embeddings_dict)} known faces")
        else:
            self.known_face_embeddings_dict = {}
            print("✅ No existing embeddings. Starting fresh.")
    
    def save_encodings(self):
        """Save embeddings to file."""
        with open(ENCODING_FILE_NAME, 'wb') as f:
            pickle.dump(self.known_face_embeddings_dict, f)
        print(f"✅ Saved embeddings for {len(self.known_face_embeddings_dict)} people")
    
    def detect_and_embed_faces(self, frame):
        """Detect faces and generate embeddings using DeepFace."""
        try:
            results = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True
            )
        except Exception as e:
            return [], [], str(e)
        
        face_locations, embeddings = [], []
        for res in results:
            if 'face' in res and res['face'] is not None:
                x = res['facial_area']['x']
                y = res['facial_area']['y']
                w = res['facial_area']['w']
                h = res['facial_area']['h']
                
                top, right, bottom, left = y, x + w, y + h, x
                face_locations.append((top, right, bottom, left))
                
                try:
                    emb = DeepFace.represent(
                        img_path=res['face'],
                        model_name=MODEL_NAME,
                        enforce_detection=False
                    )[0]['embedding']
                    embeddings.append(emb)
                except Exception as e:
                    print(f"Embedding error: {e}")
                    continue
        
        return face_locations, embeddings, None
    
    def recognize_face_robust(self, embedding):
        """Recognize face using robust matching."""
        best_match = "Unknown"
        best_distance = float('inf')
        
        for name, emb_list in self.known_face_embeddings_dict.items():
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
        
        # Require minimum samples before recognizing
        if len(self.known_face_embeddings_dict.get(best_match, [])) < MIN_SAMPLES_PER_PERSON:
            return "Unknown", best_distance
        
        if best_distance > TOLERANCE:
            return "Unknown", best_distance
        
        return best_match, best_distance
    
    def add_face(self, image, name):
        """Add a new face to the database."""
        if image is None:
            return "❌ Error: No image provided", None
        
        if not name or name.strip() == "":
            return "❌ Error: Please provide a name", None
        
        name = name.strip()
        
        face_locations, embeddings, error = self.detect_and_embed_faces(image)
        
        if error:
            return f"❌ Detection error: {error}", None
        
        if not embeddings:
            return "❌ No face detected in the image", None
        
        if len(embeddings) > 1:
            return "⚠️ Multiple faces detected. Please upload image with single face.", None
        
        # Add embedding
        if name not in self.known_face_embeddings_dict:
            self.known_face_embeddings_dict[name] = []
            log_msg = self.log_attendance(name, "Enrollment")
        else:
            log_msg = ""
        
        self.known_face_embeddings_dict[name].append(embeddings[0])
        self.save_encodings()
        
        # Draw rectangle
        result_image = image.copy()
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(result_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(result_image, f"Added: {name}", (left, top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        num_samples = len(self.known_face_embeddings_dict[name])
        remaining = max(0, MIN_SAMPLES_PER_PERSON - num_samples)
        
        status = f"✅ Successfully added sample {num_samples} for {name}\n"
        if remaining > 0:
            status += f"⚠️ Add {remaining} more sample(s) for reliable recognition\n"
        else:
            status += f"✅ {name} has enough samples for recognition\n"
        status += log_msg
        
        return status, result_image
    
    def recognize_faces(self, image):
        """Recognize faces in the input image."""
        if image is None:
            return "❌ Error: No image provided", None
        
        if len(self.known_face_embeddings_dict) == 0:
            return "❌ Error: No known faces in database. Please add faces first.", None
        
        face_locations, embeddings, error = self.detect_and_embed_faces(image)
        
        if error:
            return f"❌ Detection error: {error}", None
        
        if not embeddings:
            return "❌ No faces detected in the image", image
        
        result_image = image.copy()
        recognized_names = []
        
        for (top, right, bottom, left), emb in zip(face_locations, embeddings):
            name, distance = self.recognize_face_robust(emb)
            confidence = max(0, (1 - distance / TOLERANCE) * 100)
            
            recognized_names.append(f"{name} ({confidence:.1f}%)")
            
            # Draw rectangle and label
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(result_image, (left, top), (right, bottom), color, 2)
            cv2.rectangle(result_image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(result_image, f"{name} {confidence:.1f}%", (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Log attendance for recognized faces
            if name != "Unknown":
                self.log_attendance(name, "Attendance")
        
        result_text = f"🔍 Detected {len(embeddings)} face(s):\n" + "\n".join(f"  • {n}" for n in recognized_names)
        return result_text, result_image
    
    def get_database_info(self):
        """Get information about the current database."""
        if len(self.known_face_embeddings_dict) == 0:
            return "📭 Database is empty. No faces registered."
        
        info = f"**📊 Database Statistics**\n\n"
        info += f"Total people: {len(self.known_face_embeddings_dict)}\n\n"
        info += "**👥 Registered People:**\n\n"
        
        for name in sorted(self.known_face_embeddings_dict.keys()):
            count = len(self.known_face_embeddings_dict[name])
            status = "✅" if count >= MIN_SAMPLES_PER_PERSON else "⚠️"
            info += f"{status} **{name}**: {count} sample(s)\n"
        
        info += f"\n💡 Minimum {MIN_SAMPLES_PER_PERSON} samples required for reliable recognition"
        
        return info

# Initialize the system
fr_system = FaceRecognitionSystem()

# Create Gradio interface
with gr.Blocks(title="Face Recognition System - DeepFace") as demo:
    gr.Markdown("# 🎭 Face Recognition System (DeepFace + Google Sheets)")
    gr.Markdown(f"Using **{MODEL_NAME}** model with **{DETECTOR_BACKEND}** detector")
    
    with gr.Tabs():
        # Tab 1: Add New Face
        with gr.Tab("➕ Add New Face"):
            gr.Markdown("### Register a new face in the database")
            gr.Markdown(f"*Add at least {MIN_SAMPLES_PER_PERSON} samples per person for reliable recognition*")
            
            with gr.Row():
                with gr.Column():
                    add_image_input = gr.Image(label="Upload Image", type="numpy")
                    add_name_input = gr.Textbox(
                        label="Person's Name",
                        placeholder="Enter name...",
                        info="Enter the same name for multiple samples"
                    )
                    add_button = gr.Button("Add Face to Database", variant="primary", size="lg")
                
                with gr.Column():
                    add_output_text = gr.Textbox(label="Status", lines=5)
                    add_output_image = gr.Image(label="Processed Image")
            
            add_button.click(
                fn=fr_system.add_face,
                inputs=[add_image_input, add_name_input],
                outputs=[add_output_text, add_output_image]
            )
        
        # Tab 2: Recognize Faces
        with gr.Tab("🔍 Recognize Faces"):
            gr.Markdown("### Upload an image to recognize faces")
            
            with gr.Row():
                with gr.Column():
                    recognize_image_input = gr.Image(label="Upload Image", type="numpy")
                    recognize_button = gr.Button("Recognize Faces", variant="primary", size="lg")
                
                with gr.Column():
                    recognize_output_text = gr.Textbox(label="Recognition Results", lines=8)
                    recognize_output_image = gr.Image(label="Detected Faces")
            
            recognize_button.click(
                fn=fr_system.recognize_faces,
                inputs=[recognize_image_input],
                outputs=[recognize_output_text, recognize_output_image]
            )
        
        # Tab 3: Webcam Recognition
        with gr.Tab("📹 Webcam Recognition"):
            gr.Markdown("### Use your webcam for face recognition")
            
            with gr.Row():
                with gr.Column():
                    webcam_input = gr.Image(label="Webcam", sources=["webcam"], type="numpy")
                    webcam_button = gr.Button("Recognize from Webcam", variant="primary", size="lg")
                
                with gr.Column():
                    webcam_output_text = gr.Textbox(label="Recognition Results", lines=8)
                    webcam_output_image = gr.Image(label="Detected Faces")
            
            webcam_button.click(
                fn=fr_system.recognize_faces,
                inputs=[webcam_input],
                outputs=[webcam_output_text, webcam_output_image]
            )
        
        # Tab 4: Database Info
        with gr.Tab("📊 Database Info"):
            gr.Markdown("### View information about registered faces")
            
            db_info_button = gr.Button("Refresh Database Info", variant="primary", size="lg")
            db_info_output = gr.Markdown()
            
            db_info_button.click(
                fn=fr_system.get_database_info,
                inputs=[],
                outputs=[db_info_output]
            )
            
            demo.load(fn=fr_system.get_database_info, outputs=[db_info_output])
    
    gr.Markdown("""
    ---
    ### 📝 How to Use:
    
    1. **Add New Face**: 
       - Upload a clear, front-facing photo
       - Enter the person's name
       - Add 3+ samples for better accuracy
    
    2. **Recognize Faces**: 
       - Upload any image with faces
       - System will identify registered people
       - Attendance logged to Google Sheets
    
    3. **Webcam Recognition**: 
       - Capture photo from webcam
       - Instant face recognition
    
    ### ⚙️ Technical Details:
    - **Model**: {MODEL_NAME} (DeepFace)
    - **Detector**: {DETECTOR_BACKEND}
    - **Tolerance**: {TOLERANCE}
    - **Storage**: face_embeddings.pkl
    - **Logging**: Google Sheets (if configured)
    
    ### 📁 Required Files:
    - `face_recognition.json` - Google Sheets credentials
    - `face_embeddings.pkl` - Face encodings database
    """.format(MODEL_NAME=MODEL_NAME, DETECTOR_BACKEND=DETECTOR_BACKEND, TOLERANCE=TOLERANCE))

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)