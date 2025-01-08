
# Hackathon 2024 - Submission of Group *// Sungai Main //*

Team members:
    - // Maximilian Filling //
    - // Evander Jose Sutama //
 Wir haben zu 5 mit der Aufgabe angefangen, aber die Herausforderung bzw. die Aufgabe haben nur
 Maximilian Filling und Evander Jose Sutama erarbeitet. Die anderen 3 Gruppenmitglieder haben keine Beteiligung
 an der Aufgabe gehabt.

# Vorgehensweise:

# 1: (Datenaufbereitung)
 In einem ersten Schritt wurden die (Bauteile) rot angemalt [![Beispielbild](doc/Beispielbild.png), nur die Löcher wurden ausgelassen.
 Damit wollten wir bewirken, dass die Spiegelung der Rohdaten nicht zu einem Problem führt. Die Daten wurden entsprechend in Rohdaten
 (angemalte Bauteile) und Inputdaten (Bauteile ohne Anmalung) aufgeteilt, damit im nächsten Schritt die Masken generiert
 werden konnten. Dies erfolgte mit dem Code extract-masks.py. Somit hatten wir uns einen Datensatz von etwa 1600 Bildern
 generiert. Daraufhin wurden die Daten in train_images, train_masks, val_images und val_masks aufgeteilt. Um die
 Daten auf weitere Schritte vorzubereiten, wurden jeweils 10 Bilder in val_images und val_masks hineingegeben.
 Diese Daten kamen aus train_images und train_masks, damit eine Validierung des Trainings erfolgen konnte.

# 2: (Modell U-Net)
 Der Code implementiert ein U-Net, ein spezialisiertes Convolutional Neural Network (CNN) für Bildsegmentierung.
 Es verwendet eine Encoder-Decoder-Struktur, bei der der Encoder Bildmerkmale extrahiert und der Decoder das Bild rekonstruiert,
 unterstützt durch Skip Connections, um räumliche Details zu bewahren. Die Daten werden mit Albumentations transformiert
 und in Batches geladen. Für das Training wird BCEWithLogitsLoss verwendet, kombiniert mit dem Adam-Optimierer und
 Mixed Precision Training (AMP), um Speicher zu sparen und die Geschwindigkeit zu erhöhen. Nach jeder Epoche werden das
 Modell gespeichert, die Genauigkeit validiert und Segmentierungsergebnisse exportiert. Das U-Net ist besonders geeignet für
 präzise Segmentierungen und arbeitet effizient auch mit kleinen Datensätzen. Daraus soll eine Maske des Bauteils vorhergesagt
 werden, die 0 und 1 aufweist, damit wir den Gripper optimal positionieren können. Der Genauigkeitswert und Dice Score
 für das Modell betragen jeweils 97 %. Die Dateien, die verwendet wurden, befinden sich im Ordner solution. Somit hatten
 wir unser Modell trainiert und konnten mit der Positionierung weitermachen.



# 3: (Beste Position auf dem Bauteil)
 Der Code main.py verwendet ein trainiertes U-Net, um aus einem Eingabebild (Bauteil) eine Segmentierungsmaske zu erstellen,
 die Löcher und andere relevante Bereiche markiert. Gleichzeitig wird eine Maske des Greifers basierend auf einem Farbkanal erstellt
 (hier Rot oder Blau), um die Form des Greifers zu definieren. In unserem Code werden beide Farbkanäle automatisch berücksichtigt.
 Mit der Funktion find_valid_gripper_position wird der Greifer virtuell über das Bauteil positioniert, wobei überprüft wird,
 ob er an einer Position platziert werden kann, ohne die Löcher im Bauteil zu überdecken. Darauf folgt die Ausgabe
 in Plots, die das Bauteil und die Maske mit dem jeweiligen Greifer zeigen.

# 4: Fazit
 Das Projekt hat uns großen Spaß gemacht, da wir theoretisches Wissen mit der Praxis verknüpfen konnten. Besonders motivierend war es, 
 die gelernten Inhalte direkt anzuwenden und dadurch wertvolle Erfahrungen zu sammeln. Wir hatten die Gelegenheit, unsere Kenntnisse 
 zu vertiefen und neue Herausforderungen zu meistern. Dabei konnten wir nicht nur unser Fachwissen erweitern, sondern auch 
 unsere Teamarbeit und Problemlösungsfähigkeiten stärken. Insgesamt war es eine bereichernde Erfahrung, die uns sowohl persönlich 
 als auch fachlich weitergebracht hat.

 Vielen Dank für die Möglichkeit!

 Viele Grüße und Spaß mit unserem Modell 
 
 Maximilian Filling und Evander Jose Sutama

# Code um das Skript im Terminal auszuführen:
```
python solution/main.py --csv_path evaluate/task.csv
``
 

