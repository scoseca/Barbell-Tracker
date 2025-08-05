from ultralytics import YOLO
import os

if __name__ == '__main__':
    # Aiuta a gestire la frammentazione della memoria
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Carica un modello più leggero
    model = YOLO('yolov8m.pt')  # Medium
    
    # Addestra con impostazioni ad alte prestazioni
    results = model.train(
        data='E:/Tesi/Barbel-Tracker/data/dataset yoloV8 /data.yaml',
        epochs=150,
        imgsz=1024,                 # Alta risoluzione
        batch=16,                   # Batch size abbastanza grande
        patience=30,
        project='barbell_tracker',
        name='train_highutil',
        save=True,
        device='0',
        workers=8,
        amp=True,
        cache='disk',
        mosaic=1.0,
        mixup=0.15,
        close_mosaic=10,
    )

    print("Addestramento completato!")