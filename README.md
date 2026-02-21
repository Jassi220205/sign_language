🖐️ Real-Time ASL Sign Language Recognition System

A real-time American Sign Language (ASL) recognition system using MediaPipe for hand detection and YOLO classification for gesture recognition.

The system detects hand landmarks, extracts a hand region of interest (ROI), performs classification using a trained YOLO model, and applies temporal smoothing for stable predictions.

📌 Project Status

✅ Draft 1 Completed

✅ Real-time webcam inference implemented

⚠ Currently detects only: Q, Y, and 0

🔄 Evaluation metrics being computed using eval_asl.py

🚧 Draft 2 will include full performance analysis

🏗️ System Architecture
Webcam Input
      ↓
MediaPipe Hand Detection
      ↓
Bounding Box Extraction
      ↓
YOLO Classification (best.pt)
      ↓
Confidence Filtering
      ↓
Temporal Smoothing (Deque Buffer)
      ↓
Real-Time Display + FPS
📂 Project Structure
SIGN/
│
├── asl_live.py        # Real-time ASL detection (Draft 1)
├── eval_asl.py        # Evaluation script (metrics calculation)
├── train.py           # Training script
├── data.yaml          # Dataset configuration
├── requirements.txt   # Dependencies
└── README.md
⚙️ Installation
1️⃣ Clone Repository
git clone <your-repo-url>
cd SIGN
2️⃣ Create Virtual Environment (Recommended)
python -m venv asl_env
asl_env\Scripts\activate   # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Run Real-Time Detection
python asl_live.py

Press Q to exit.


🔬 Current Limitations

Model currently recognizes only a subset of ASL signs.

Dataset imbalance may affect generalization.

Sensitive to lighting conditions.

Confusion observed between visually similar gestures.

🚀 Future Work (Draft 2)

Full multi-class evaluation

Ablation study on smoothing window

Hyperparameter tuning

Improved dataset balancing

Deployment using Gradio or Streamlit

Model optimization for lightweight inference

📦 Dependencies

Python 3.9+

OpenCV

MediaPipe

Ultralytics YOLO

NumPy

scikit-learn

tqdm

Install all using:

pip install -r requirements.txt
📌 Note on Model Weights

If best.pt is not included in this repository:

Download it separately

Place inside:

asl_runs/asl_yolo_cls/weights/
👩‍💻 Author

Jassmitha Jammu
ASL Sign Language Recognition Project
