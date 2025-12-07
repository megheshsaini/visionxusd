# 🎭 Face Recognition System

Computer Vision final team project at USD using DeepFace for robust face recognition submitted by Group 11:

1. Meghesh Saini
2. Aishwarya Gulhane
3. Dhrub Satyam

Under the guidance of Amit Butail.

## 🚀 Features

- Face detection and recognition using DeepFace (Facenet model)
- Web interface using Gradio
- Google Sheets integration for attendance logging
- Support for multiple face samples per person
- Real-time webcam recognition

## 📦 Installation
```bash
# Clone the repository
[git clone https://github.com/YOUR_USERNAME/visionxusd.git](https://github.com/megheshsaini/visionxusd.git)
cd visionxusd

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Setup

1. **Google Sheets Setup** (Optional - for attendance logging):
   - Create a Google Sheet named `facerecognitionapp`
   - Create a service account and download JSON credentials
   - Rename the JSON file to `face_recognition.json`
   - Place it in the project root directory
   - Share the Google Sheet with the service account email

2. **Run the application**:
```bash
   python gradio_app.py
```

3. **Access the interface**:
   - Open browser at `http://localhost:7860`

## 📖 Usage

### Adding New Faces
1. Go to "➕ Add New Face" tab
2. Upload a clear photo
3. Enter the person's name
4. Add at least 3 samples per person for best accuracy

### Recognizing Faces
1. Go to "🔍 Recognize Faces" tab
2. Upload an image
3. Click "Recognize Faces"
4. View results with confidence scores

### Webcam Recognition
1. Go to "📹 Webcam Recognition" tab
2. Capture photo from webcam
3. Click "Recognize Faces"

## 🔧 Configuration

Edit these settings in `gradio_app.py`:

- `TOLERANCE = 0.25` - Recognition threshold (lower = stricter)
- `MIN_SAMPLES_PER_PERSON = 3` - Minimum samples needed
- `MODEL_NAME = "Facenet"` - Face recognition model
- `DETECTOR_BACKEND = "retinaface"` - Face detection method

## 📁 Files

- `gradio_app.py` - Main application
- `face_embeddings.pkl` - Face database (auto-generated)
- `face_recognition.json` - Google credentials (not included)

## ⚠️ Important Notes

- **DO NOT** commit `face_recognition.json` to GitHub (contains private keys)
- Add at least 3 face samples per person for reliable recognition
- First run will download AI models (~100MB)

## 🛠️ Technologies

- DeepFace (Facenet model)
- RetinaFace detector
- Gradio web interface
- Google Sheets API
- OpenCV

## 📄 License

under USD Computer Vision Course Project (Guidance of Amit Butail)
