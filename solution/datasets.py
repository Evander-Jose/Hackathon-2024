import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class HackathonDataset(Dataset):
    def __init__(self, mask_dir, image_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Liste der Dateien filtern, um unerwünschte Dateien auszuschließen
        self.images = [file for file in os.listdir(image_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
    """
           Konstruktor der Klasse HackathonDataset.
           Initialisiert den Datensatz mit Pfaden zu Bildern und Masken sowie optionalen Transformationen.

           Args:
               mask_dir (str): Pfad zum Masken-Verzeichnis.
               image_dir (str): Pfad zum Bild-Verzeichnis.
               transform (callable, optional): Transformationsfunktion für Datenaugmentation. Standardwert ist None.
    """
    def __len__(self):
        return len(self.images)

    """ Gibt die Anzahl der Bilder im Datensatz zurück.
        Returns:
               int: Anzahl der Bilder im Verzeichnis.
    """

    def __getitem__(self, index):
        # Pfad zum Bild und zur entsprechenden Maske
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        """
               Lädt das Bild und die zugehörige Maske basierend auf dem angegebenen Index.
               Führt optional Transformationen durch.

               Args:
                   index (int): Index des gewünschten Elements.

               Returns:
                   tuple: (Bild, Maske), wobei das Bild ein RGB-Bild (NumPy-Array) und die Maske ein Graustufen-Bild ist.
        """
        # Bild laden und in RGB konvertieren
        image = np.array(Image.open(img_path).convert("RGB"))
        # Maske laden und in Graustufen konvertieren
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # Maskenwerte umskalieren (255 -> 1.0)
        mask[mask == 255.0] = 1.0

        # Falls Transformationen vorhanden sind, auf Bild und Maske anwenden
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        # Bild und Maske zurückgeben
        return image, mask

