import os
import shutil
import random
import yaml

print("Script avviato...")

# Directory di base
original_images_dir = "E:/Tesi/Barbel-Tracker/data/extracted_frames/dataset2"  
labels_dir = "E:/Tesi/Barbel-Tracker/data/dataset yoloV8/labels/train"  
output_dir = "E:/Tesi/Barbel-Tracker/data/dataset yoloV8 2"  

print(f"Controllando directories:")
print(f"- Immagini: {original_images_dir} (esiste: {os.path.exists(original_images_dir)})")
print(f"- Etichette: {labels_dir} (esiste: {os.path.exists(labels_dir)})")

# Crea la struttura
os.makedirs(f"{output_dir}/images/train", exist_ok=True)
os.makedirs(f"{output_dir}/images/val", exist_ok=True)
os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
os.makedirs(f"{output_dir}/labels/val", exist_ok=True)


# Elenca tutte le etichette
label_files = [f for f in os.listdir("E:/Tesi/Barbel-Tracker/data/set2/labels/train") if f.endswith('.txt')]
random.shuffle(label_files)  # Mescola per un split casuale
print(f"Trovati {len(label_files)} file di etichette")

# Split: 80% train, 20% val
split_idx = int(len(label_files) * 0.8)
train_labels = label_files[:split_idx]
val_labels = label_files[split_idx:]

# Copia i file train
for label_file in train_labels:
    image_base = os.path.splitext(label_file)[0]
    
    # Trova l'immagine corrispondente (controlla diverse estensioni)
    for ext in ['.jpg', '.jpeg', '.png']:
        img_path = os.path.join(original_images_dir, f"{image_base}{ext}")
        if os.path.exists(img_path):
            shutil.copy(img_path, f"{output_dir}/images/train/{image_base}{ext}")
            shutil.copy(os.path.join("E:/Tesi/Barbel-Tracker/data/set2/labels/train", label_file), 
                       f"{output_dir}/labels/train/{label_file}")
            break

# Copia i file val
for label_file in val_labels:
    image_base = os.path.splitext(label_file)[0]
    
    # Trova l'immagine corrispondente
    for ext in ['.jpg', '.jpeg', '.png']:
        img_path = os.path.join(original_images_dir, f"{image_base}{ext}")
        if os.path.exists(img_path):
            shutil.copy(img_path, f"{output_dir}/images/val/{image_base}{ext}")
            shutil.copy(os.path.join("E:/Tesi/Barbel-Tracker/data/set2/labels/train", label_file), 
                       f"{output_dir}/labels/val/{label_file}")
            break

# Crea il file data.yaml
data_yaml = {
    'train': './images/train',
    'val': './images/val',
    'nc': 1,
    'names': ['barbell']
}

with open(f"{output_dir}/data.yaml", 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"Dataset organizzato in {output_dir}")