import cv2
import os
import glob
import argparse
import hashlib

def extract_frames(video_dir, output_dir, fps=1, max_frames=None):
    """
    Estrai frame da tutti i video in una directory
    
    Args:
        video_dir: Directory contenente i video
        output_dir: Directory dove salvare i frame
        fps: Quanti frame estrarre al secondo
        max_frames: Numero massimo di frame da estrarre per video (None = nessun limite)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova tutti i file video
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
    
    print(f"Trovati {len(video_files)} file video")
    
    total_saved = 0
    
    for video_file in video_files:
        # Genera un ID univoco per il video basato sul nome file
        video_id = hashlib.md5(os.path.basename(video_file).encode()).hexdigest()[:8]
        
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Errore nell'apertura di {video_file}")
            continue
        
        # Calcola l'intervallo tra frame da estrarre
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nElaborazione di {os.path.basename(video_file)}")
        print(f"FPS originale: {video_fps:.2f}, frame totali: {total_frames}")
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Nome file univoco con ID video
                output_path = os.path.join(output_dir, f"{video_id}_{saved_count:06d}.jpg")
                cv2.imwrite(output_path, frame)
                saved_count += 1
                total_saved += 1
                
                if saved_count % 10 == 0:
                    print(f"Salvati {saved_count} frame da questo video", end="\r")
                    
                # Limita il numero di frame se specificato
                if max_frames and saved_count >= max_frames:
                    print(f"\nRaggiunto limite di {max_frames} frame per questo video")
                    break
            
            frame_count += 1
        
        cap.release()
        print(f"\nCompletato {os.path.basename(video_file)}: salvati {saved_count} frame")
    
    print(f"\nProcesso completato! Salvati in totale {total_saved} frame in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrai frame da video per dataset")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory contenente i video")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory dove salvare i frame")
    parser.add_argument("--fps", type=float, default=1, help="Frame per secondo da estrarre")
    parser.add_argument("--max_frames", type=int, default=None, help="Massimo numero di frame per video")
    
    args = parser.parse_args()
    extract_frames(args.video_dir, args.output_dir, args.fps, args.max_frames)