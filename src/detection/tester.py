from ultralytics import YOLO
import argparse
import os

def test_model(path_immagine, path_video, model_path):
    # Verifica che il file del modello esista
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Il file del modello non esiste: {model_path}")
    
    # Carica il modello
    print(f"Caricamento modello da: {model_path}")
    model = YOLO(model_path, task="detect")
    
    # Test su immagini
    print(f"Elaborazione immagine: {path_immagine}")
    results = model(path_immagine, save=True, conf=0.25)
    print(f"Risultato immagine salvato in: {results[0].save_dir}")

    # Test su video con tracking
    print(f"Elaborazione video: {path_video}")
    try:
        results = model.track(
            path_video,
            save=True, 
            tracker="bytetrack.yaml",  # Algoritmo di tracking
            conf=0.25,                 # Soglia di confidenza
            show=True                  # Mostra durante l'elaborazione
        )
        print("Elaborazione video completata")
    except Exception as e:
        print(f"Errore durante l'elaborazione del video: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test del modello YoloV8 per immagini e video')
    parser.add_argument('--path_immagine', required=True, help='Percorso dell\'immagine di input')
    parser.add_argument('--path_video', required=True, help='Percorso del video di input')
    parser.add_argument('--model_path', default='barbell_tracker/train_highutil/weights/best.pt', 
                        help='Percorso al file del modello (default: barbell_tracker/train_highutil/weights/best.pt)')
    
    args = parser.parse_args()
    test_model(args.path_immagine, args.path_video, args.model_path)