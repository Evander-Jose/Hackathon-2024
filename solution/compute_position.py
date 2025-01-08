import os.path
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import argparse
import pandas as pd
from solution_utils import find_valid_gripper_position,visualize_gripper_position, add_row_to_csv, extract_mask

from model import UNET
from solution_utils import extract_mask, find_valid_gripper_position

# Checkpoint-Pfad für das trainierte Modell
checkpoint_path = "model/my_checkpoint_wbce.pth.tar"


def compute_position(part_image_path: Path, gripper_image_path: Path) -> tuple[float, float, float]:
    """
       Berechnet die optimale Position und Orientierung:
        :param part_image_path: Path to the part image
        :param gripper_image_path: Path to the gripper image
        :return: The x, y and angle of the gripper. If a valid position cannot be found, it returns (-42, -42, -42)
    """

    #Maske für den Greifer erstellen
    gripper = cv.imread(gripper_image_path)                               # Greifer-Bild laden
    gripper_mask = extract_mask(gripper, threshold=250)             # Erstellen einer Binärmaske basierend auf Farbwerten

    # Modell initialisieren und mit den trainierten Parametern laden
    model = UNET(in_channels=3, out_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])

    # Vorhersage mit dem Modell machen
    image = Image.open(part_image_path).convert("RGB")                    # Bild des Bauteils laden und in RGB konvertieren
    original_image = np.array(image)                                # Originalbild in NumPy-Array konvertieren (für Visualisierung)
    # Transformationen vorbereiten: Umwandlung in Tensor und Normalisierung
    transform = transforms.Compose([
        transforms.ToTensor(),                                      # Umwandeln in Tensor
        transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])  # Optional: Normalisierung
    ])
    # Transformationen auf das Bild anwenden
    if transform:
        image = transform(image)
    image = image.unsqueeze(0)                                      # Batch-Dimension hinzufügen

    # Modell in den Evaluationsmodus setzen
    model.eval()
    with torch.no_grad():                                           # Gradientenberechnung deaktivieren
        workpiece_mask = torch.sigmoid(model(image))                # Modellvorhersage (Wahrscheinlichkeiten)
        workpiece_mask = (workpiece_mask > 0.5).float()             # Binäre Maske erzeugen
        workpiece_mask = workpiece_mask.numpy()                     # In NumPy-Format konvertieren
        workpiece_mask = workpiece_mask[0, 0, :, :]                 # Batch- und Kanal-Dimension entfernen

    # Invertiere die Maske (falls nötig)
    workpiece_mask = 1 - workpiece_mask

    # Gültige Greifer-Position anhand der Masken ermitteln
    result = find_valid_gripper_position(workpiece_mask, gripper_mask)

    # Visualisiere die Greifer-Position auf der Bauteil-Maske
    visualize_gripper_position(workpiece_mask, gripper_mask, result, original_image, name=f"{os.path.basename(part_image_path)}+{os.path.basename(gripper_image_path)}")

    # Ergebnis (x, y, alpha) speichern, wenn ein gültiger Punkt gefunden wurde
    if result is not None:
        x, y, alpha = result
        return x, y, alpha
    else:
        return -42, -42, -42

