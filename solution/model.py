import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
               Doppelte Convolution: Zwei aufeinanderfolgende Convolutional-BatchNorm-ReLU-Operationen.
               Args:
                   in_channels (int): Eingabekanäle.
                   out_channels (int): Ausgabekanäle.
               """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # Erste Conv
            nn.BatchNorm2d(out_channels),                                                       # Batch Normalization
            nn.ReLU(inplace=True),                                                              # ReLU-Aktivierung
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),# Zweite Conv
            nn.BatchNorm2d(out_channels),                                                       # Batch Normalization
            nn.ReLU(inplace=True),                                                              # ReLU-Aktivierung
        )

    def forward(self, x):
        """
                Vorwärtsdurchlauf: Anwenden des Conv-Blocks auf die Eingabe.
                Args:
                    x: Eingabetensor.
                Returns:
                    Tensor nach doppelter Convolution.
                """
        return self.conv(x)

# UNET-Klasse für das Modell
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        """
                UNET-Architektur: Besteht aus Downsampling, Bottleneck und Upsampling.
                Args:
                    in_channels (int): Anzahl der Eingabekanäle.
                    out_channels (int): Anzahl der Ausgabekanäle.
                    features (list): Anzahl der Kanäle in jeder Schicht.
                """
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()                                              # Listenspeicher für Up-Konstruktionen
        self.downs = nn.ModuleList()                                            # Listenspeicher für Down-Konstruktionen
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                       # Max-Pooling-Operation

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))                 # Hinzufügen von DoubleConv-Blöcken
            in_channels = feature                                               # Aktualisieren der Eingabekanäle für die nächste Schicht

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2) # Transponierte Convolution
            )
            self.ups.append(DoubleConv(feature * 2, feature))                   # Hinzufügen von DoubleConv-Blöcken
        # Bottleneck-Schicht
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        # Letzte Convolution (zu den gewünschten Ausgabekanälen)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
                Vorwärtsdurchlauf des Modells.
                Args:
                    x: Eingabetensor.
                Returns:
                    Ausgabetensor mit vorhergesagten Masken.
                """
        skip_connections = []                                                  # Liste für Skip-Connections (für den Encoder-Decoder-Übergang)

        # Encoder: Downsampling mit Speicherung der Skip-Connections
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)                                          # Speichern der Zwischenwerte
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]                               # Reihenfolge der Skip-Connections umkehren

        # Decoder: Upsampling mit Skip-Connections
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)                                                # Transponierte Convolution
            skip_connection = skip_connections[idx // 2]                        # Entsprechende Skip-Connection

            # Falls die Formen nicht übereinstimmen, anpassen
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Concatenation von Upsampling und Skip-Connection
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)                                 # Weitere DoubleConv-Operation

        return self.final_conv(x)                                              # Endgültige Vorhersage

# Test-Funktion außerhalb der Klasse
def test():
    """
        Testet die UNET-Architektur mit Dummy-Daten.
        Erstellt eine zufällige Eingabe und überprüft, ob die Ausgabeform korrekt ist.
        """
    x = torch.randn((3, 1, 160, 160))                                          # Batch Size 3, 1 Channel, 160x160 Input Size
    model = UNET(in_channels=1, out_channels=1)                                # Modell mit 1 Eingabe- und 1 Ausgabekanal
    preds = model(x)                                                           # Modellvorhersage
    print(f"Predicted Shape: {preds.shape}")
    print(f"Input Shape: {x.shape}")
    assert preds.shape == x.shape, "Shapes do not match!"                      # Überprüfung der Form

# Test-Aufruf
if __name__ == "__main__":
    test()
