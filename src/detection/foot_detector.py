import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
class FootPositionDetector:
    """
    Classe per determinare la posizione del piede usando MediaPipe.
    """
    
    def __init__(self):
        """Inizializzazione del rilevatore."""
        # Indici dei landmark per i piedi in MediaPipe Pose
        self.LEFT_HEEL = 29
        self.RIGHT_HEEL = 30
        self.LEFT_FOOT_INDEX = 31  # Punta del piede sinistro
        self.RIGHT_FOOT_INDEX = 32  # Punta del piede destro
    
    def extract_foot_positions_from_video(self, video_path, pose_landmarker, n_clusters=3, visualize=True):
        """
        Analizza l'intero video per trovare le posizioni più stabili dei piedi.
        
        Args:
            video_path: Percorso del video da analizzare
            pose_landmarker: Istanza del pose_landmarker di MediaPipe già inizializzata
            n_clusters: Numero di cluster per il raggruppamento delle posizioni (default: 3)
            visualize: Se True, visualizza i cluster trovati
            
        Returns:
            Posizione del piede più vicino alla camera come (x, y)
        """
        # Apri il video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossibile aprire il video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Liste per memorizzare le posizioni dei piedi
        left_foot_positions = []
        right_foot_positions = []
        
        # Elabora il video
        pbar = tqdm(total=frame_count, desc="Analisi posizioni dei piedi")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Converti frame per MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Rileva la posa
            pose_result = pose_landmarker.detect(mp_image)
            
            # Se c'è una posa rilevata, estrai le posizioni dei piedi
            if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                landmarks = pose_result.pose_landmarks[0]
                
                # Estrai posizioni (normalizzate)
                left_heel = (landmarks[self.LEFT_HEEL].x, landmarks[self.LEFT_HEEL].y)
                right_heel = (landmarks[self.RIGHT_HEEL].x, landmarks[self.RIGHT_HEEL].y)
                left_toe = (landmarks[self.LEFT_FOOT_INDEX].x, landmarks[self.LEFT_FOOT_INDEX].y)
                right_toe = (landmarks[self.RIGHT_FOOT_INDEX].x, landmarks[self.RIGHT_FOOT_INDEX].y)
                
                # Converti in coordinate pixel
                left_midfoot = (
                    int((left_heel[0] + left_toe[0]) / 2 * width),
                    int((left_heel[1] + left_toe[1]) / 2 * width)
                )
                right_midfoot = (
                    int((right_heel[0] + right_toe[0]) / 2 * width),
                    int((right_heel[1] + right_toe[1]) / 2 * width)
                )
                
                # Salva le posizioni
                left_foot_positions.append(left_midfoot)
                right_foot_positions.append(right_midfoot)
            
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # Verifica che ci siano abbastanza posizioni rilevate
        if len(left_foot_positions) < 10 or len(right_foot_positions) < 10:
            print("Avvertimento: Rilevate poche posizioni dei piedi. I risultati potrebbero non essere affidabili.")
            # Usa una posizione predefinita se ci sono troppi pochi dati
            if len(left_foot_positions) > 0:
                return np.mean(left_foot_positions, axis=0)
            elif len(right_foot_positions) > 0:
                return np.mean(right_foot_positions, axis=0)
            else:
                return (width // 2, height // 2)  # Centro dell'immagine
        
        # Determina quale piede è più vicino alla camera
        left_x_mean = np.mean([pos[0] for pos in left_foot_positions])
        right_x_mean = np.mean([pos[0] for pos in right_foot_positions])
        
        # Scegliamo il piede che è più verso il bordo dell'inquadratura
        if abs(left_x_mean) < abs(right_x_mean - width):
            print("Utilizzo del piede sinistro per l'analisi (più vicino alla camera)")
            foot_positions = np.array(left_foot_positions)
        else:
            print("Utilizzo del piede destro per l'analisi (più vicino alla camera)")
            foot_positions = np.array(right_foot_positions)
        
        # Applica clustering K-means per trovare le posizioni più comuni
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(foot_positions)
        
        # Trova il cluster più popolato
        labels = kmeans.labels_
        counts = np.bincount(labels)
        most_common_cluster = np.argmax(counts)
        cluster_center = kmeans.cluster_centers_[most_common_cluster]
        
        # Visualizza i risultati se richiesto
        if visualize:
            plt.figure(figsize=(10, 8))
            colors = ['r', 'g', 'b', 'y', 'c', 'm']  # Colori per i diversi cluster
            
            for i in range(n_clusters):
                cluster_points = foot_positions[labels == i]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                           c=colors[i % len(colors)], label=f'Cluster {i+1} ({counts[i]} punti)')
            
            # Evidenzia i centri dei cluster
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                       s=200, c='k', marker='X', label='Centri dei cluster')
            
            # Evidenzia il cluster più comune
            plt.scatter(cluster_center[0], cluster_center[1], 
                       s=300, c='lime', marker='*', label='Posizione scelta')
            
            plt.title('Clustering delle posizioni del piede')
            plt.xlabel('Posizione X (pixel)')
            plt.ylabel('Posizione Y (pixel)')
            plt.legend()
            plt.grid(True)
            
            # Salva il grafico
            plt.savefig('foot_position_clusters.png', dpi=300, bbox_inches='tight')
            print("Grafico del clustering salvato come 'foot_position_clusters.png'")
        
        print(f"Posizione del midfoot determinata: ({cluster_center[0]:.2f}, {cluster_center[1]:.2f})")
        return tuple(cluster_center)
