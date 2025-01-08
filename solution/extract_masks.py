import numpy
import glob
import cv2 as cv
import matplotlib.pyplot as plt

# Pfad: hier liegen die Rohdaten, wurden entsprechend mit Rot angemalt.
base_path = "/Users/max/Documents/Rohdaten"
#Ursprüngliche Rohdaten der Bauteile. Nicht angemalt.
#base_path = "/Users/max/PycharmProjects/Hackathon-2024/dataset/input"

# Sammelt alle PNG-Dateipfade aus Unterverzeichnissen
all_images_paths = glob.glob(f"{base_path}/*/*.png")

# Zielverzeichnis: Hier werden die Masken und Bauteile gespeichert.
save_base_path = "/Users/max/PycharmProjects/Hackathon-2024/dataset/train_masks"
#save_base_path = "/Users/max/PycharmProjects/Hackathon-2024/dataset/train_images"



# Iteration über alle gesammelten Bilderpfade
for index, image_path in enumerate(all_images_paths):

    # Extrahiert den Dateinamen aus dem Pfad
    splited = image_path.split("/")[-1]

    # Überprüft, ob der Dateiname mit "mask" oder "mirr" beginnt
    if splited[:4] == "mask" or splited[:4] == "mirr":

        # Bild laden (im BGR-Format)
        image = cv.imread(image_path)

        # Maske erstellen: Bedingungen für Farbkanäle
        mask_b = image[:, :, 2] >= 215
        mask_r = image[:, :, 1] <= 200
        mask_g = image[:, :, 0] <= 200

        # Kombinieren der Maskenbedingungen
        total = mask_r * 1 + mask_b * 1 + mask_g * 1
        total = total / 3
        mask = total >= 1

        # Maske in 8-Bit (0-255) umwandeln
        mask = (mask * 255).astype('uint8')

        # Speichern der Datei mit numerischem Namen
        image_name = f"{save_base_path}/{index}.png"

        # Maske im Zielverzeichnis speichern
        cv.imwrite(image_name, mask)   # wird in 2 unterschiedliche Pfade gespeichert

        # Debug-Ausgabe
        #print(f"Gespeichert: {image_name}")

        #plt.imshow(image)
        #plt.show()
        #plt.imshow(mask, cmap="grey")
        #plt.show()
