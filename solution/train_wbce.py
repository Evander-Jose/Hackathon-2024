import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET

from utils import (
load_checkpoint,
save_checkpoint,
get_loaders,
check_accuracy,                                             # Funktion zur Überprüfung der Modellgenauigkeit
save_predictions_as_imgs,                                   # Funktion zum Speichern der Modellvorhersagen als Bilder
 )

# Hyperparameter etc.
LEARNING_RATE = 1e-4                                        # Lernrate für den Optimierer
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu" # Gerät auswählen (Apple M1/MPS oder CPU)
BATCH_SIZE = 32
NUM_EPOCHS = 100                                            # Anzahl der Trainings-Epochen
NUM_WORKERS = 2                                             # Anzahl der Datenlade-Worker
IMAGE_HEIGHT = 160                                          # 1280 originally
IMAGE_WIDTH = 240                                           # 1918 originally
PIN_MEMORY = True                                           # Optimiert den Speicherzugriff (schnellerer Datenzugriff)
LOAD_MODEL = False                                          # Gibt an, ob ein gespeichertes Modell geladen werden soll
TRAIN_IMG_DIR = "../dataset/train_images/"                  # Verzeichnis mit Trainingsbildern
TRAIN_MASK_DIR = "../dataset/train_masks/"                  # Verzeichnis mit Trainingsmasken
VAL_IMG_DIR = "../dataset/val_images/"
VAL_MASK_DIR = "../dataset/val_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    """
        Trainiert das Modell für eine Epoche.
        Args:
            loader: DataLoader mit Trainingsdaten.
            model: Das UNET-Modell.
            optimizer: Optimierungsalgorithmus (z. B. Adam).
            loss_fn: Verlustfunktion (z. B. BCEWithLogitsLoss).
            scaler: GradScaler für Mixed Precision Training.
        """

    for batch_idx, (data, targets) in enumerate(loop):
        # Daten und Ziele auf das Gerät verschieben (CPU/GPU)
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Vorwärtsdurchlauf
        with torch.cuda.amp.autocast():                     # Mixed Precision Training
            predictions = model(data)                       # Modellvorhersagen
            loss = loss_fn(predictions, targets)            # Verlust berechnen

        # backward
        optimizer.zero_grad()                               # Gradienten zurücksetzen
        scaler.scale(loss).backward()                       # Verlust skalieren und Rückpropagation durchführen
        scaler.step(optimizer)                              # Optimierer aktualisieren
        scaler.update()                                     # Skalierung aktualisieren

        # Fortschrittsanzeige aktualisieren
        loop.set_postfix(loss=loss.item())


def main():
    """
       Führt den Trainings- und Validierungsprozess aus.
       1. Transformationen definieren.
       2. Modell, Verlustfunktion und Optimierer initialisieren.
       3. Datenlademechanismen einrichten.
       4. Trainings- und Validierungsschleifen ausführen.
       """
    # Transformationen für Trainingsdaten (Augmentierungen)
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),   # Skalierung der Bilder
            A.Rotate(limit=35, p=1.0),                          # Rotieren um maximal 35 Grad
            A.HorizontalFlip(p=0.5),                            # Horizontales Spiegeln
            A.VerticalFlip(p=0.1),                              # Vertikales Spiegeln
            A.Normalize(                                        # Normalisierung der Farbkanäle
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(), # Umwandlung in PyTorch-Tensor
        ],
    )
    # Transformationen für Validierungsdaten (keine Augmentierung)
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),   # Skalierung
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),                                       # Umwandlung in Tensor
        ],
    )
    # UNET-Modell initialisieren
    model =UNET(in_channels=3, out_channels=1).to(DEVICE)

    # Verlustfunktion und Optimierer definieren
    loss_fn = nn.BCEWithLogitsLoss()                            # Binary Cross Entropy Loss mit Logits
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam-Optimierer

    # Datenlademechanismen einrichten
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    # GradScaler für Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler()
    # Trainingsschleife
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler) # Eine Epoche trainieren

        # Modell speichern
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }

        save_checkpoint(checkpoint, filename="../model/UNET.pth.tar") # Checkpoint speichern

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")


if __name__ == "__main__":
    main()


