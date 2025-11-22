import numpy as np
import matplotlib.pyplot as plt
import math

# INITIALISATION
A = plt.imread('Polytech.jpg')
x, y, z = A.shape[:3]
x_new = (x // 8) * 8
y_new = (y // 8) * 8

img = A[:x_new, :y_new, :z]
img2 = img.astype(np.int16) - 128

# Pour garder une seule couleur (canal bleu)
img2[:, :, 0] = 0
img2[:, :, 1] = 0

print(f"l'image : {img2}")
plt.imshow(np.clip(img2 + 128, 0, 255).astype(np.uint8))  # uint8: 0 to 255
plt.show()

B = np.array(img2[:, :, 2])
P = np.zeros((8, 8))

for k in range(8):
    for l in range(8):
        if k == 0:
            C = 1 / (math.sqrt(2))
        else:
            C = 1
        P[k][l] = (1 / 2) * C * math.cos(((2 * l + 1) * k * math.pi) / 16)

Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 13, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

x2, y2 = B.shape

# COMPRESSION
def compression_bloc(M, P):
    P_inv = np.transpose(P)
    D = P @ M @ P_inv  # Changement de base
    M2 = np.floor(np.divide(D, Q)).astype(int)  # Quantification
    return M2

M_compressée = np.zeros((x2, y2))

for x in range(0, x2, 8):
    for y in range(0, y2, 8):
        bloc = B[x:x+8, y:y+8]
        new_bloc = compression_bloc(bloc, P)
        M_compressée[x:x+8, y:y+8] = new_bloc

nb_nonzeros = np.count_nonzero(B)  # Nombre de coefficients non-nuls dans la matrice initiale
nb_nonzeros_2 = np.count_nonzero(M_compressée)  # Nombre de coefficients non-nuls dans la matrice compressée
erreur = (nb_nonzeros_2 / nb_nonzeros) * 100  # Erreur en %
taux_de_compression = 100 - erreur  # Taux de compression en %

print(f"Taux de compression: {taux_de_compression}")
print(f"Erreur: {erreur}")
