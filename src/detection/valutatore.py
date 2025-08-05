from ultralytics import YOLO

if __name__ == '__main__':
    # Carica il modello addestrato (best.pt contiene i pesi migliori)
    model = YOLO('E:/Tesi/Barbel-Tracker/src/detection/barbell_tracker/train_finetuned/weights/best.pt') 
    # Esegui la validazione
    results = model.val()
    print(f"mAP50: {results.box.map50:.4f}")  # Precisione media con IoU 0.5
    print(f"mAP50-95: {results.box.map:.4f}")  # Precisione media con IoU da 0.5 a 0.95