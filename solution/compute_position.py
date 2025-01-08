import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from model import UNET
import torch
from PIL import Image
from torchvision import transforms
import argparse
import pandas as pd
from utils import find_valid_gripper_position,visualize_gripper_position, add_row_to_csv, extract_mask


# Argumente parsen (CSV-Pfad wird über Kommandozeilenargumente angegeben)
parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", type=str, help="path/to/task.png")
args = parser.parse_args()

# Checkpoint-Pfad für das trainierte Modell
checkpoint_path = "model/my_checkpoint_wbce.pth.tar"


def create_output(csv_file_path: str):
    """
       Hauptfunktion, die den gesamten Workflow abdeckt:
       1. Liest die Eingabedaten aus einer CSV-Datei.
       2. Führt Masken-Generierung und Modell-Inferenz durch.
       3. Findet gültige Greifer-Positionen und schreibt die Ergebnisse in eine Ausgabe-CSV.
       """

    # Dummy-Ausgabedatei erstellen mit den gewünschten Spalten
    headers_output = ["part", "gripper", "x", "y", "alpha"]
    data_output = pd.DataFrame(columns=headers_output)
    file_path_output = "evaluate/output.csv"
    data_output.to_csv(file_path_output, index=False)

    # CSV-Datei einlesen (enthält Informationen über Bauteile und Greifer)
    data = pd.read_csv(csv_file_path)
    parts = data["part"]                                                # Spalte mit Pfaden zu den Bauteilen
    grippers = data["gripper"]                                          # Spalte mit Pfaden zu den Greifern

    # Iteration über jedes Bauteil und dessen zugehörigen Greifer
    for part_path, gripper_path in zip(parts, grippers):

        #Maske für den Greifer erstellen
        gripper = cv.imread(gripper_path)                               # Greifer-Bild laden
        gripper_mask = extract_mask(gripper, threshold=250)             # Erstellen einer Binärmaske basierend auf Farbwerten

        # Modell initialisieren und mit den trainierten Parametern laden
        model = UNET(in_channels=3, out_channels=1)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])

        # Vorhersage mit dem Modell machen
        image = Image.open(part_path).convert("RGB")                    # Bild des Bauteils laden und in RGB konvertieren
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
            workpiece_mask = workpiece_mask.numpy()                     #In NumPy-Format konvertieren
            workpiece_mask = workpiece_mask[0, 0, :, :]                 # Batch- und Kanal-Dimension entfernen

        # Invertiere die Maske (falls nötig)
        workpiece_mask = 1 - workpiece_mask
        plt.imshow(workpiece_mask, cmap="grey")
        plt.show()

        # Gültige Greifer-Position anhand der Masken ermitteln
        result = find_valid_gripper_position(workpiece_mask, gripper_mask)

        # Visualisiere die Greifer-Position auf der Bauteil-Maske
        visualize_gripper_position(workpiece_mask, gripper_mask, result, original_image)

        # Ergebnis (x, y, alpha) speichern, wenn ein gültiger Punkt gefunden wurde
        if result is not None:
            x, y, alpha = result
            csv_result = (part_path, gripper_path, x,y, alpha)

            # Ergebnisse in die Ausgabedatei hinzufügen
            add_row_to_csv(file_path_output, csv_result)

# Einstiegspunkt des Skripts
if __name__ == "__main__":
    create_output(csv_file_path=args.csv_path)

