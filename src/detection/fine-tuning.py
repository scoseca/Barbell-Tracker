from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('E:/Tesi/Barbel-Tracker/src/detection/barbell_tracker/train_highutil/weights/best.pt')

    # Esegui il fine-tuning sul nuovo dataset con i casi "banali"
    results = model.train(
        data='E:/Tesi/Barbel-Tracker/data/dataset yoloV8 2/data.yaml',
        epochs=50,                              # Meno epoche per il fine-tuning
        imgsz=896,                              # Dimensione delle immagini
        batch=16,                               # Batch size
        patience=15,                            # Early stopping
        project='barbell_tracker',              # Nome progetto
        name='train_finetuned',                 # Nome per questo training
        save=True,                              # Salva i risultati
        device='0',                             # GPU
        amp=True,                               # Mixed precision
        exist_ok=True                           # Sovrascrivi la cartella se esiste
    ) 