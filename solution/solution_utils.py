import torch
import torchvision
import os
from datasets import HackathonDataset
from torch.utils.data import DataLoader
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import pandas as pd

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
        Speichert den aktuellen Modellzustand und Optimiererparameter in einer Datei.
        Args:
            state (dict): Enthält Modell- und Optimiererzustand.
            filename (str): Name der Datei, in die der Zustand gespeichert wird.
        """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """
       Lädt einen gespeicherten Modellzustand in das angegebene Modell.
       Args:
           checkpoint (dict): Gespeicherte Parameter (state_dict).
           model (nn.Module): Das Modell, in das die Parameter geladen werden.
       """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
    ):
    """
       Erstellt DataLoader für Trainings- und Validierungsdaten.
       Args:
           train_dir (str): Pfad zu Trainingsbildern.
           train_maskdir (str): Pfad zu Trainingsmasken.
           val_dir (str): Pfad zu Validierungsbildern.
           val_maskdir (str): Pfad zu Validierungsmasken.
           batch_size (int): Anzahl der Bilder pro Batch.
           train_transform (callable): Transformationen für Trainingsdaten.
           val_transform (callable): Transformationen für Validierungsdaten.
           num_workers (int): Anzahl der Subprozesse, die Daten laden.
           pin_memory (bool): Optimierung für schnellen Speicherzugriff.
       Returns:
           tuple: DataLoader für Trainings- und Validierungsdaten.
       """
# Trainingsdatensatz
    train_ds = HackathonDataset(
    image_dir=train_dir,
    mask_dir=train_maskdir,
    transform=train_transform,
        )

    train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=True,                               # Zufälliges Durchmischen der Trainingsdaten
        )

    val_ds = HackathonDataset(
    image_dir=val_dir,
    mask_dir=val_maskdir,
    transform=val_transform,
        )

    val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=False                               # Keine Durchmischung bei Validierung
        )

    return train_loader, val_loader

def check_accuracy(loader, model, device="mps"):
    """
        Überprüft die Genauigkeit des Modells auf den Validierungsdaten.
        Args:
            loader (DataLoader): DataLoader mit Validierungsdaten.
            model (nn.Module): Das trainierte Modell.
            device (str): Zielgerät (CPU/GPU/MPS).
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()                                # Setze Modell in den Evaluationsmodus

    with torch.no_grad():                       # Deaktiviert Gradientenberechnung
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))     # Vorhersagen durch Sigmoid skalieren
            preds = (preds > 0.5).float()       # Binärmaske erzeugen
            num_correct += (preds == y).sum()   # Korrekte Pixel zählen
            num_pixels += torch.numel(preds)    # Gesamtanzahl der Pixel
            dice_score += (2 *(preds * y).sum()) / ((preds + y).sum() + 1e-8)


    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()                               # Setze Modell zurück in den Trainingsmodus


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="mps"):
    """
        Speichert die Vorhersagen des Modells und die Ground-Truth-Masken als Bilder.
        Args:
            loader (DataLoader): DataLoader mit Validierungsdaten.
            model (nn.Module): Das trainierte Modell.
            folder (str): Zielordner für die gespeicherten Bilder.
            device (str): Zielgerät (CPU/GPU/MPS).
    """

        # Sicherstellen, dass der Speicherordner existiert
    os.makedirs(folder, exist_ok=True)

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            preds = torch.sigmoid(model(x))         # Modellvorhersagen
            preds = (preds > 0.5).float()           # Binärmaske erzeugen

        # Speichere die vorhergesagten Masken
        pred_file = f"{folder}/pred_{idx}.png"
        torchvision.utils.save_image(preds, pred_file)

        # Speichere die Ground-Truth-Masken (y)
        gt_file = f"{folder}/gt_{idx}.png"          # Ground-Truth-Dateiname
        torchvision.utils.save_image(y.unsqueeze(1), gt_file)

    model.train()

def find_valid_gripper_position(component_mask, gripper_mask):
    """
       Findet eine gültige Greiferposition auf der Komponente.
       Args:
           component_mask (np.ndarray): Maske der Komponente.
           gripper_mask (np.ndarray): Maske des Greifers.
       Returns:
           tuple: (x, y, alpha) für die Greiferposition und -rotation, None falls keine gefunden wird.
       """
        # Code für die Suche nach einer gültigen Greiferposition...
    component_h, component_w = component_mask.shape
    gripper_h, gripper_w = gripper_mask.shape

        # Ensure the masks are binary
    component_mask = (component_mask > 0).astype(np.uint8)
    gripper_mask = (gripper_mask > 0).astype(np.uint8)

        # Create a padded version of the component mask to handle edge cases
    padding = max(gripper_h, gripper_w)
    padded_component = np.pad(component_mask, pad_width=padding, mode='constant', constant_values=0)

    center_x, center_y = component_w // 2, component_h // 2  # Mitte der Komponente

        # Versuche zuerst die Mitte zu platzieren
    for angle in range(0, 360, 5):                           # Test rotation angles in steps of 5 degrees
        rotated_gripper = rotate(gripper_mask, angle, reshape=True, order=1)

        # Find bounding box of rotated gripper
        gripper_nonzero = np.argwhere(rotated_gripper)
        if gripper_nonzero.size == 0:
            continue
        min_y, min_x = gripper_nonzero.min(axis=0)
        max_y, max_x = gripper_nonzero.max(axis=0)
        rotated_gripper_cropped = rotated_gripper[min_y:max_y + 1, min_x:max_x + 1]

        gripper_h, gripper_w = rotated_gripper_cropped.shape

        # Teste die zentrale Position
        x_start = center_x - gripper_w // 2 + padding
        y_start = center_y - gripper_h // 2 + padding

        # Den Bereich von Interesse (ROI) aus der gepolsterten Komponente extrahieren
        roi = padded_component[y_start:y_start + gripper_h, x_start:x_start + gripper_w]

        # Überprüfen, ob eine Überlappung zwischen der Greifer-Maske und den Löchern (Komponente == 1) besteht.
        if not np.any((rotated_gripper_cropped == 1) & (roi == 1)):
            return x_start - padding, y_start - padding, angle

        # Wenn Mitte nicht möglich, suche andere Positionen
    for angle in range(0, 360, 5):                      # Wiederhole für andere Positionen
        rotated_gripper = rotate(gripper_mask, angle, reshape=True, order=1)

        # Finde die Begrenzungsbox des rotierten Greifers.
        gripper_nonzero = np.argwhere(rotated_gripper)
        if gripper_nonzero.size == 0:
            continue
        min_y, min_x = gripper_nonzero.min(axis=0)
        max_y, max_x = gripper_nonzero.max(axis=0)
        rotated_gripper_cropped = rotated_gripper[min_y:max_y + 1, min_x:max_x + 1]

        gripper_h, gripper_w = rotated_gripper_cropped.shape

        for y in range(padding, padding + component_h - gripper_h):
            for x in range(padding, padding + component_w - gripper_w):
                roi = padded_component[y:y + gripper_h, x:x + gripper_w]

                # Check for overlap between gripper mask and holes (component == 1)
                if not np.any((rotated_gripper_cropped == 1) & (roi == 1)):
                    return x - padding, y - padding, angle

        # Wenn keine gültige Position gefunden wird
    return None



def visualize_gripper_position(component_mask, gripper_mask, result, original_image):
    """
       Visualisiert die Greiferposition auf der Komponente.
       Args:
           component_mask (np.ndarray): Maske der Komponente.
           gripper_mask (np.ndarray): Maske des Greifers.
           result (tuple): (x, y, alpha) von find_valid_gripper_position.
           original_image (np.ndarray): Originalbild der Komponente.
       """
    # Code für die Visualisierung der Greiferposition...
    if result is None:
        print("No valid position found.")
        return

    x, y, alpha = result

    # Rotieren Sie die Greifermaske um den angegebenen Winkel
    rotated_gripper = rotate(gripper_mask, alpha, reshape=True, order=1)

    # Create a visualization
    plt.figure(figsize=(12, 8))

    # Überlagern Sie die Greifermaske auf der Komponentenmaske
    visualization = np.copy(component_mask).astype(float)
    gripper_overlay = np.copy(original_image).astype(float)
    for i in range(rotated_gripper.shape[0]):
        for j in range(rotated_gripper.shape[1]):
            if rotated_gripper[i, j] > 0:
                vi, vj = y + i, x + j
                if 0 <= vi < visualization.shape[0] and 0 <= vj < visualization.shape[1]:
                    visualization[vi, vj] = 0.5  # Mark gripper area
                    gripper_overlay[vi, vj] = [255, 0, 0]  # Mark gripper area in red

    # Zeigen Sie die Komponentenmaske mit der überlagerten Greifermaske
    plt.subplot(1, 2, 1)
    plt.imshow(visualization, cmap='gray')
    plt.title("Component Mask with Gripper")

    # Zeigen Sie das Originalbild mit der überlagerten Greifermaske
    plt.subplot(1, 2, 2)
    plt.imshow(gripper_overlay.astype(np.uint8))
    plt.title(f"Original Image with Gripper")

    # Add text for x, y, alpha on the plot
    plt.gcf().text(0.5, 0.01, f"Gripper Position: x={x}, y={y}, alpha={alpha}°", ha='center', fontsize=12)

    plt.show()

def add_row_to_csv(file_path_output, new_row):
    """
        Fügt eine neue Zeile zur Ausgabedatei (CSV) hinzu.
        Args:
            file_path_output (str): Pfad zur CSV-Datei.
            new_row (tuple): Neue Zeile mit den Werten.
        """

    try:
        data_output = pd.read_csv(file_path_output) # Bestehende Datei laden
    except FileNotFoundError:
        data_output = pd.DataFrame(columns=["workpiece", "gripper", "x", "y", "alpha"])

    data_output.loc[len(data_output)] = new_row # Neue Zeile hinzufügen
    data_output.to_csv(file_path_output, index=False) # Datei speichern

def extract_mask(image, threshold=250):
    """
    Extrahiert eine Binärmaske aus einem RGB-Bild, unabhängig davon, in welchem Farbkanal sich die Maske befindet.

    Args:
        image (np.ndarray): Eingabebild im RGB-Format.
        threshold (int): Schwellenwert, ab dem Pixel als Maske erkannt werden.

    Returns:
        np.ndarray: Binärmaske (0 = Hintergrund, 1 = Maske).
    """
    # Prüfe, ob das Bild überhaupt Kanäle hat
    if len(image.shape) < 3 or image.shape[2] != 3:
        raise ValueError("Das Bild muss ein RGB-Bild mit 3 Kanälen sein.")

    # Initialisiere eine leere Maske
    binary_masks = []

    # Iteriere über alle drei Kanäle
    for i in range(3):
        channel_mask = (image[:, :, i] >= threshold).astype(np.uint8)  # Schwellenwert anwenden
        binary_masks.append(channel_mask)

    # Kombiniere die Masken (logisches ODER)
    combined_mask = np.max(binary_masks, axis=0)

    return combined_mask

