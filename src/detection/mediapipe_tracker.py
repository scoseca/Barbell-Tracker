import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class BarbellTrackerLandmarker:
    def __init__(self, pose_landmarker):      
        self.pose_landmarker = pose_landmarker

        # Definisci connessioni per la visualizzazione
        self.pose_connections = mp.solutions.pose.POSE_CONNECTIONS
        
        # Mappa dei landmark per gli angoli articolari
        self.landmarks_dict = {
            'NOSE': 0, 'LEFT_EYE_INNER': 1, 'LEFT_EYE': 2, 'LEFT_EYE_OUTER': 3,
            'RIGHT_EYE_INNER': 4, 'RIGHT_EYE': 5, 'RIGHT_EYE_OUTER': 6,
            'LEFT_EAR': 7, 'RIGHT_EAR': 8, 'MOUTH_LEFT': 9, 'MOUTH_RIGHT': 10,
            'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12, 'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
            'LEFT_WRIST': 15, 'RIGHT_WRIST': 16, 'LEFT_PINKY': 17, 'RIGHT_PINKY': 18,
            'LEFT_INDEX': 19, 'RIGHT_INDEX': 20, 'LEFT_THUMB': 21, 'RIGHT_THUMB': 22,
            'LEFT_HIP': 23, 'RIGHT_HIP': 24, 'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28, 'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
            'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32
        }
        
        # Joint pairs per calcolare angoli
        self.joint_pairs = {
            'right_knee': ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'],
            'right_hip': ['RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'],
            'right_elbow': ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
        }
        
        # Storage per dati di biomeccanica
        self.barbell_trajectory = []
        self.joint_trajectories = {}
        self.timestamps = []
        self.joint_angles = {joint: [] for joint in self.joint_pairs}
    
    def calculate_angle(self, a, b, c):
        """Calcola l'angolo ABC in gradi"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def process_frame(self, frame, timestamp):
        frame_height, frame_width = frame.shape[:2]
        output_frame = frame.copy()

        # Rileva la posa con MediaPipe Pose Landmarker
        # Converti l'immagine per MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Rileva pose
        pose_result = self.pose_landmarker.detect(mp_image)
        
        # Estrai e visualizza i landmark se ci sono pose rilevate
        if pose_result.pose_landmarks:
            pose_landmarks = pose_result.pose_landmarks[0]  # Prendi la prima posa
            
            # Disegna i landmark e le connessioni
            self._draw_landmarks_on_image(output_frame, pose_landmarks)
            
            # Estrai coordinate normalizzate e convertile in pixel
            landmarks_px = {}
            for landmark_name, idx in self.landmarks_dict.items():
                if idx < len(pose_landmarks):
                    landmark = pose_landmarks[idx]
                    x = landmark.x * frame_width
                    y = landmark.y * frame_height
                    z = landmark.z  # La profondità z è già normalizzata
                    landmarks_px[landmark_name] = (x, y, z)
                    
                    # Traccia le articolazioni nel tempo
                    if landmark_name not in self.joint_trajectories:
                        self.joint_trajectories[landmark_name] = []
                    self.joint_trajectories[landmark_name].append((x, y))
            
            # Calcola gli angoli articolari
            for joint_name, landmarks_names in self.joint_pairs.items():
                if all(name in landmarks_px for name in landmarks_names):
                    a = landmarks_px[landmarks_names[0]][:2]  # Solo x,y
                    b = landmarks_px[landmarks_names[1]][:2]
                    c = landmarks_px[landmarks_names[2]][:2]
                    
                    angle = self.calculate_angle(a, b, c)
                    self.joint_angles[joint_name].append(angle)
                    
                    # Annota l'angolo sul frame
                    cv2.putText(output_frame, f"{joint_name}: {angle:.1f}°", 
                              (int(b[0]), int(b[1]) - 15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # # 3. Relazione tra bilanciere e articolazioni
            # if barbell_position is not None and 'RIGHT_WRIST' in landmarks_px:
            #     # Calcola distanza tra bilanciere e polso destro
            #     wrist_pos = landmarks_px['RIGHT_WRIST'][:2]  # Solo x,y
            #     distance = np.sqrt((barbell_position[0] - wrist_pos[0])**2 + 
            #                       (barbell_position[1] - wrist_pos[1])**2)
                
            #     # Annota la distanza sul frame
            #     cv2.putText(output_frame, f"Wrist-Bar: {distance:.1f}px", 
            #               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
            #     # Disegna la linea che connette il polso al bilanciere
            #     cv2.line(output_frame, 
            #            (int(wrist_pos[0]), int(wrist_pos[1])), 
            #            (int(barbell_position[0]), int(barbell_position[1])),
            #            (0, 255, 255), 1)
        
        return output_frame
    
    def _draw_landmarks_on_image(self, image, landmarks):
        """Disegna i landmark della posa sull'immagine"""
        height, width = image.shape[:2]
        
        # Disegna i punti dei landmark
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(image, (x, y), 5, (0, 128, 255), -1)
        
        # Disegna le connessioni
        connections = self.pose_connections
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                
                start_x = int(start_point.x * width)
                start_y = int(start_point.y * height)
                end_x = int(end_point.x * width)
                end_y = int(end_point.y * height)
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), (64, 224, 208), 2)
    
    def analyze_video(self, video_path, output_path=None):
        """Analizza un video completo"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Configurazione output video
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Timestamp del frame in secondi
            timestamp = frame_count / fps
            
            # Elabora il frame
            annotated_frame = self.process_frame(frame, timestamp)
            
            # Mostra il risultato
            cv2.imshow("Combined Barbell Analysis", annotated_frame)
            
            # Salva il frame elaborato
            if output_path:
                out.write(annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
# Esempio di utilizzo
if __name__ == "__main__":
    
    tracker = BarbellTrackerLandmarker('barbell_tracker/train_highutil/weights/best.pt')
    tracker.analyze_video("path/to/your/video.mp4", "output_landmarker.mp4")