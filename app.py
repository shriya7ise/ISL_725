import cv2  # For webcam capture
import pickle  # For loading the model
import json  # For JSON response from Flask route
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import mediapipe as mp  # For MediaPipe Holistic

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(static_image_mode=False)

app = Flask(__name__)

try:
    with open('cnn.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: Model file not found.")
    exit() 

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(frame_rgb)

    annotated_frame = frame.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        )

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

    processed_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    return processed_frame

def process_webcam_video(frames):
    processed_frames = []
    for frame in frames:
        processed_frame = process_frame(frame)
        processed_frames.append(processed_frame)

    prediction = model.predict(processed_frames)
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    return render_template('live_recog.html')

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({'prediction': 'Unable to access webcam'})

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    features = [process_frame(frame) for frame in frames]
    predictions = model.predict(features) if features else "No gesture detected"
    return jsonify({'prediction': predictions[0] if isinstance(predictions, list) else predictions})

if __name__ == '__main__':
    app.run(debug=True)
    
    
    