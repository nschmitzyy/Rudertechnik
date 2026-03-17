import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

# --- KONFIGURATION ---
st.set_page_config(page_title="Rowing Tech Analyzer", layout="wide")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """Berechnet den Winkel zwischen drei Punkten (Vektor-Mathematik)."""
    a = np.array(a) # Erstes Gelenk
    b = np.array(b) # Scheitelpunkt
    c = np.array(c) # Drittes Gelenk
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- UI DESIGN ---
st.title("🚣‍♂️ Rowing Tech Analyzer")
st.markdown("""
Analysiere deine Rudertechnik auf dem Ergometer. 
1. Lade ein Video von der **Seite** hoch.
2. Die KI erkennt deine Gelenke und analysiert die Sequenz (Beine-Körper-Arme).
""")

uploaded_file = st.sidebar.file_uploader("Video hochladen (mp4, mov)", type=['mp4', 'mov'])

if uploaded_file is not None:
    # Temporäres Speichern des Videos
    tfile = open("temp_video.mp4", "wb")
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture("temp_video.mp4")
    stframe = st.empty()
    
    data_log = []
    
    # --- PROCESSING LOOP ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Konvertierung für MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Koordinaten extrahieren (Beispiel linke Seite)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Winkel berechnen
            knee_angle = calculate_angle(hip, knee, ankle)
            hip_angle = calculate_angle(shoulder, hip, knee)
            arm_angle = calculate_angle(shoulder, elbow, wrist)
            
            # Daten für Diagramm speichern
            data_log.append({
                "Knie": knee_angle,
                "Hüfte": hip_angle,
                "Arme": arm_angle
            })

            # --- FEEDBACK LOGIK (BEISPIEL) ---
            feedback = "Technik OK"
            color = (0, 255, 0) # Grün
            
            # Prüfung: Ziehen die Arme zu früh im Drive?
            if knee_angle < 140 and arm_angle < 160:
                feedback = "FEHLER: Arme ziehen zu früh!"
                color = (0, 0, 255) # Rot
            
            # Zeichnen auf dem Frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
        stframe.image(frame, channels="BGR", use_container_width=True)

    cap.release()

    # --- ANALYSE-DIAGRAMM ---
    if data_log:
        st.subheader("Analyse der Bewegungsphasen")
        df = pd.DataFrame(data_log)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['Knie'], name="Kniewinkel", line=dict(color='blue')))
        fig.add_trace(go.Scatter(y=df['Hüfte'], name="Hüftwinkel", line=dict(color='green')))
        fig.add_trace(go.Scatter(y=df['Arme'], name="Armwinkel", line=dict(color='red')))
        
        fig.update_layout(title="Winkelverlauf über das Video", xaxis_title="Frames", yaxis_title="Winkel in Grad")
        st.plotly_chart(fig)

st.sidebar.info("Tipp: Achte auf eine gute Beleuchtung und eine Kamera-Position direkt auf Hüfthöhe des Ruderers.")
