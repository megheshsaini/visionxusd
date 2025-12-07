"""
Face Recognition System with DeepFace and Google Sheets
Using Streamlit for web interface
"""

import streamlit as st
import cv2
import numpy as np
import pickle
import os
import time
import gspread
from deepface import DeepFace
from PIL import Image

# ---------------- CONFIG ----------------
SPREADSHEET_NAME = "facerecognitionapp"
WORKSHEET_NAME = "Attendance"
SERVICE_ACCOUNT_FILE = "face_recognition.json"
ENCODING_FILE_NAME = "face_embeddings.pkl"
DETECTOR_BACKEND = "retinaface"
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
            st.warning(f"⚠️ {SERVICE_ACCOUNT_FILE} not found. Google Sheets logging disabled.")
            return
        
        try:
            gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
            spreadsheet = gc.open(SPREADSHEET_NAME)
            self.worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
            st.success(f"✅ Connected to Google Sheet '{SPREADSHEET_NAME}'")
        except gspread.SpreadsheetNotFound:
            st.error(f"❌ Spreadsheet '{SPREADSHEET_NAME}' not found")
        except gspread.WorksheetNotFound:
            spreadsheet = gc.open(SPREADSHEET_NAME)
            self.worksheet = spreadsheet.add_worksheet(title=WORKSHEET_NAME, rows="1000", cols="3")
            self.worksheet.append_row(["Timestamp", "Person Name", "Log Type"])
            st.success(f"✅ Worksheet '{WORKSHEET_NAME}' created")
        except Exception as e:
            st.warning(f"⚠️ Google Sheets error: {e}")
    
    def log_attendance(self, name, log_type="Attendance"):
        """Log attendance or enrollment into Google Sheet."""
        if self.worksheet:
            try:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                self.worksheet.insert_row([timestamp, name, log_type], 2)
                return f"✅ Logged: {name} ({log_type})"
            except Exception as e:
                return f"⚠️ Logging failed: {e}"
        return None
    
    def load_encodings(self):
        """Load embeddings from file."""
        if os.path.exists(ENCODING_FILE_NAME):
            with open(ENCODING_FILE_NAME, 'rb') as f:
                self.known_face_embeddings_dict = pickle.load(f)
        else:
            self.known_face_embeddings_dict = {}
    
    def save_encodings(self):
        """Save embeddings to file."""
        with open(ENCODING_FILE_NAME, 'wb') as f:
            pickle.dump(self.known_face_embeddings_dict, f)
    
    def detect_and_embed_faces(self, image_array):
        """Detect faces and generate embeddings using DeepFace."""
        try:
            results = DeepFace.extract_faces(
                img_path=image_array,
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
        
        if len(self.known_face_embeddings_dict.get(best_match, [])) < MIN_SAMPLES_PER_PERSON:
            return "Unknown", best_distance
        
        if best_distance > TOLERANCE:
            return "Unknown", best_distance
        
        return best_match, best_distance
    
    def add_face(self, image, name):
        """Add a new face to the database."""
        image_array = np.array(image)
        
        face_locations, embeddings, error = self.detect_and_embed_faces(image_array)
        
        if error:
            return None, f"❌ Detection error: {error}"
        
        if not embeddings:
            return None, "❌ No face detected in the image"
        
        if len(embeddings) > 1:
            return None, "⚠️ Multiple faces detected. Use image with single face."
        
        # Add embedding
        if name not in self.known_face_embeddings_dict:
            self.known_face_embeddings_dict[name] = []
            log_msg = self.log_attendance(name, "Enrollment")
        else:
            log_msg = None
        
        self.known_face_embeddings_dict[name].append(embeddings[0])
        self.save_encodings()
        
        # Draw rectangle
        result_image = image_array.copy()
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(result_image, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(result_image, f"Added: {name}", (left, top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        num_samples = len(self.known_face_embeddings_dict[name])
        remaining = max(0, MIN_SAMPLES_PER_PERSON - num_samples)
        
        status = f"✅ Successfully added sample {num_samples} for **{name}**\n\n"
        if remaining > 0:
            status += f"⚠️ Add **{remaining}** more sample(s) for reliable recognition\n"
        else:
            status += f"✅ **{name}** has enough samples for recognition\n"
        
        if log_msg:
            status += f"\n{log_msg}"
        
        return result_image, status
    
    def recognize_faces(self, image):
        """Recognize faces in the input image."""
        image_array = np.array(image)
        
        face_locations, embeddings, error = self.detect_and_embed_faces(image_array)
        
        if error:
            return None, f"❌ Detection error: {error}"
        
        if not embeddings:
            return image_array, "❌ No faces detected in the image"
        
        result_image = image_array.copy()
        recognized_info = []
        
        for (top, right, bottom, left), emb in zip(face_locations, embeddings):
            name, distance = self.recognize_face_robust(emb)
            confidence = max(0, (1 - distance / TOLERANCE) * 100)
            
            recognized_info.append((name, confidence))
            
            # Draw rectangle and label
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(result_image, (left, top), (right, bottom), color, 3)
            cv2.rectangle(result_image, (left, bottom - 40), (right, bottom), color, cv2.FILLED)
            cv2.putText(result_image, f"{name} {confidence:.1f}%", (left + 6, bottom - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Log attendance
            if name != "Unknown":
                self.log_attendance(name, "Attendance")
        
        status = f"🔍 **Detected {len(embeddings)} face(s):**\n\n"
        for name, conf in recognized_info:
            status += f"• **{name}** ({conf:.1f}% confidence)\n"
        
        return result_image, status

# Initialize Streamlit
st.set_page_config(
    page_title="Face Recognition - DeepFace",
    page_icon="🎭",
    layout="wide"
)

# Initialize session state
if 'fr_system' not in st.session_state:
    with st.spinner("Initializing Face Recognition System..."):
        st.session_state.fr_system = FaceRecognitionSystem()

# Header
st.title("🎭 Face Recognition System")
st.markdown(f"**Model:** {MODEL_NAME} | **Detector:** {DETECTOR_BACKEND}")

# Sidebar
with st.sidebar:
    st.header("📊 Statistics")
    num_people = len(st.session_state.fr_system.known_face_embeddings_dict)
    
    total_samples = sum(len(v) for v in st.session_state.fr_system.known_face_embeddings_dict.values())
    st.metric("Registered People", num_people)
    st.metric("Total Samples", total_samples)
    
    st.markdown("---")
    st.header("Navigation")
    page = st.radio(
        "Select Page:",
        ["➕ Add New Face", "🔍 Recognize Faces", "📹 Webcam", "📊 Database Info"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown(f"""
    ### ⚙️ Settings
    - **Min Samples**: {MIN_SAMPLES_PER_PERSON}
    - **Tolerance**: {TOLERANCE}
    - **Storage**: `{ENCODING_FILE_NAME}`
    """)

# Add New Face Page
if page == "➕ Add New Face":
    st.header("Register a New Face")
    st.info(f"💡 Add at least **{MIN_SAMPLES_PER_PERSON}** samples per person for reliable recognition")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear front-facing photo"
        )
        
        name = st.text_input(
            "Person's Name",
            placeholder="Enter name...",
            help="Use the same name for multiple samples"
        )
        
        if st.button("➕ Add Face to Database", type="primary", use_container_width=True):
            if uploaded_file and name:
                image = Image.open(uploaded_file)
                
                with st.spinner("Processing..."):
                    result_image, message = st.session_state.fr_system.add_face(image, name.strip())
                
                if result_image is not None:
                    with col2:
                        st.success("Face Added!")
                        st.markdown(message)
                        st.image(result_image, caption="Processed Image", use_container_width=True)
                else:
                    st.error(message)
            else:
                st.warning("⚠️ Please upload an image and enter a name")
    
    with col2:
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

# Recognize Faces Page
elif page == "🔍 Recognize Faces":
    st.header("Recognize Faces from Image")
    
    if len(st.session_state.fr_system.known_face_embeddings_dict) == 0:
        st.warning("⚠️ No faces in database. Please add faces first.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload any image with faces"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("🔍 Recognize Faces", type="primary", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        result_image, message = st.session_state.fr_system.recognize_faces(image)
                    
                    with col2:
                        if result_image is not None:
                            st.markdown(message)
                            st.image(result_image, caption="Detected Faces", use_container_width=True)
                        else:
                            st.error(message)

# Webcam Page
elif page == "📹 Webcam":
    st.header("Webcam Face Recognition")
    
    if len(st.session_state.fr_system.known_face_embeddings_dict) == 0:
        st.warning("⚠️ No faces in database. Please add faces first.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            camera_image = st.camera_input("Take a picture")
            
            if camera_image:
                if st.button("🔍 Recognize Faces", type="primary", use_container_width=True):
                    image = Image.open(camera_image)
                    
                    with st.spinner("Analyzing..."):
                        result_image, message = st.session_state.fr_system.recognize_faces(image)
                    
                    with col2:
                        if result_image is not None:
                            st.markdown(message)
                            st.image(result_image, caption="Detected Faces", use_container_width=True)
                        else:
                            st.error(message)

# Database Info Page
elif page == "📊 Database Info":
    st.header("Database Information")
    
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()
    
    if len(st.session_state.fr_system.known_face_embeddings_dict) == 0:
        st.info("📭 Database is empty. No faces registered.")
    else:
        st.markdown(f"### 👥 Registered People ({len(st.session_state.fr_system.known_face_embeddings_dict)})")
        
        for name in sorted(st.session_state.fr_system.known_face_embeddings_dict.keys()):
            count = len(st.session_state.fr_system.known_face_embeddings_dict[name])
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                status = "✅" if count >= MIN_SAMPLES_PER_PERSON else "⚠️"
                st.markdown(f"{status} **{name}**")
            
            with col2:
                st.metric("Samples", count)
            
            with col3:
                if count >= MIN_SAMPLES_PER_PERSON:
                    st.success("Ready")
                else:
                    st.warning("Need more")
        
        st.markdown("---")
        st.info(f"💡 Minimum **{MIN_SAMPLES_PER_PERSON}** samples required for reliable recognition")

# Footer
st.markdown("---")
st.markdown("""
### 📝 Instructions:
1. **Add New Face**: Upload 3+ clear photos of each person
2. **Recognize Faces**: Upload any image to identify people
3. **Webcam**: Use camera for real-time recognition
4. **Database Info**: View all registered faces

### 📁 Required Files:
- `face_recognition.json` - Google Sheets credentials
- `face_embeddings.pkl` - Created automatically
""")
