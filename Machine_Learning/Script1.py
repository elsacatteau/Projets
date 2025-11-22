# TODO: Import necessary libraries
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.decomposition import PCA
from skimage.filters import sobel
from sklearn.preprocessing import MinMaxScaler
##########################################
## Data loading and first visualisation
##########################################
digits=load_digits()

# Load the handwritten digits dataset
def nombre_aleatoire(): #pour afficher un nombre aléatoire

    nombre=random.randint(0,9) #on génère un nombre aléatoire compris entre 0 et 9
    print("L'image générée correspond au nombre", nombre)

    digits=load_digits()
    plt.matshow(digits.images[nombre], cmap="gray")
    plt.axis('off')
    plt.show()


def nombre_aleatoire_i(nombre): #pour afficher un nombre i
    print("L'image générée correspond au nombre", nombre)
    digits=load_digits()
    plt.matshow(digits.images[nombre], cmap="gray")
    plt.axis('off')
    plt.show()

# Visualize some images - graph the first4 images from the data base
for i in range(4):
    nombre_aleatoire_i(i)

# Display at least one random sample par class (some repetitions of class... oh well)
def plot_multi(data, y):
    nplots = 16
    nb_classes = len(np.unique(y))
    cur_class = 0
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        to_display_idx = np.random.choice(np.where(y == cur_class)[0])
        plt.imshow(data[to_display_idx].reshape((8,8)), cmap='binary')
        plt.title(cur_class)
        plt.axis('off')
        cur_class = (cur_class + 1) % nb_classes
    plt.show()

plot_multi(digits.data, digits.target)

##########################################
## Data exploration and first analysis
##########################################

def get_statistics_text(targets):
    # TODO: Write your code here, returning at least the following useful infos:
    # * Label names
    # * Number of elements per class
    labels, counts = np.unique(targets, return_counts=True)
    return labels, counts

# TODO: Call the previous function and generate graphs and prints for exploring and visualising the database
labels,counts=get_statistics_text(digits.target)
plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color='blue', edgecolor='black')
plt.xlabel("Chiffre")
plt.ylabel("Nombre d'occurences")
plt.title("Répartition des classes dans le dataset")
plt.xticks(labels)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

##########################################
## Start data preprocessing
##########################################

# Access the whole dataset as a matrix where each row is an individual (an image in our case)
# and each column is a feature (a pixel intensity in our case)
## X = [
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 1 as a row
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 2 as a row
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 3 as a row
#  [Pixel1, Pixel2, ..., Pixel64]   # Image 4 as a row
#]

# TODO: Create a feature matrix and a vector of labels
X = digits.data
y = digits.target

# Print dataset shape
print(f"Feature matrix shape: {X.shape}. Max value = {np.max(X)}, Min value = {np.min(X)}, Mean value = {np.mean(X)}")
print(f"Labels shape: {y.shape}")


# TODO: Normalize pixel values to range [0,1]
F = X/16.0  # Feature matrix after scaling

# Print matrix shape
print(f"Feature matrix F shape: {F.shape}. Max value = {np.max(F)}, Min value = {np.min(F)}, Mean value = {np.mean(F)}")

##########################################
## Dimensionality reduction
##########################################


### just an example to test, for various number of PCs
sample_index = 0
original_vector = F[sample_index] #notre vecteur original en 64 dimensions
original_image = F[sample_index].reshape(8, 8)  #matrice 8x8 pour la visualisation


# TODO: Using the specific sample above, iterate the following:
# * Generate a PCA model with a certain value of principal components
# * Compute the approximation of the sample with this PCA model
# * Reconstruct a 64 dimensional vector from the reduced dimensional PCA space
# * Reshape the resulting approximation as an 8x8 matrix
# * Quantify the error in the approximation

fig, axes = plt.subplots(4,4,figsize=(10,10))
fig.suptitle('Image originale et approximation par ACP', fontsize=16)

for i in range(16):
   ax = axes[i // 4, i % 4]

   if i==0: #on affiche l'image originale
       ax.imshow(original_image, cmap='binary')
       ax.set_title("Original")
       ax.axis('off')

   else:
       # On applique PCA avec i composantes
       pca = PCA(n_components=i)
       F_reduced = pca.fit_transform(F)
       F_approx = pca.inverse_transform(F_reduced)

       # On récupère la reconstruction du vecteur original
       reconstructed_vector = F_approx[sample_index]
       reconstructed_image = reconstructed_vector.reshape(8, 8)

       # On calcule l'erreur (MSE)
       error = np.mean((original_vector - reconstructed_vector) ** 2)

       # On affiche l'image reconstruite
       ax.imshow(reconstructed_image, cmap='binary')
       ax.set_title(f"{i} comps\nErr={error:.4f}")
       ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()

#### TODO: Expolore the explanined variance of PCA and plot
pca= PCA()
pca.fit(F)

#On récupère la variance expliquée pour chaque composante
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

#On trace la variance expliquée et la variance cumulée
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, label="Variance expliquée (par composante)")
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, color='red', marker='o', label="Variance cumulée")
plt.xlabel("Nombre de composantes principales")
plt.ylabel("Variance expliquée")
plt.title("Variance expliquée par les composantes principales (PCA)")
plt.xticks(np.arange(1, 65, step=4))
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

#Remarque : pour savoir combien de composantes est ce qu'il faut pour capturer au moins de 95% de la variance, on peut écrire le code suivant :
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Nombre de composantes pour expliquer au moins 95% de la variance : {n_components_95}")

#Distribution des valeurs propres

eigenvalues = pca.explained_variance_ # on récupère les valeurs propres (la variance expliquée brute)

# Tracer la distribution des valeurs propres
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-', color='purple')
plt.title("Distribution des valeurs propres (variance absolue capturée par chaque composante)")
plt.xlabel("Composante principale")
plt.ylabel("Valeur propre (variance)")
plt.grid(True)
plt.tight_layout()
plt.show()

#Observations
#On observe une forte décroissance au début, ce qui indique que les 10 premières composantes capturent > 90% de la variance. Par ailleurs, on remarque que les dernières valeurs propres tendent vers 0, ce qui signifie que certaines composantes n'apportent aucune information, et qu'elles peuvent donc être ignorées sans perte significative.

#De plus, en général, les valeurs propres sont assez faibles (< 0.7), ce qui confirme que les images des chiffres manuscrits sont fortement redondantes


### TODO: Display the whole database in 2D:
#Pour aller plus loin le faire en 3D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(F) # F est ta matrice normalisée (entre 0 et 1)
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=15)
plt.colorbar()
plt.title("Visualisation 2D après PCA")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.grid(True)
plt.show()

#Remarque : tracé en 3D

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=3)
X_pca_3d = pca.fit_transform(F)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap='tab10', s=20, alpha=0.8)

ax.set_title("Visualisation 3D après PCA")
ax.set_xlabel("Composante 1")
ax.set_ylabel("Composante 2")
ax.set_zlabel("Composante 3")

cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
cbar.set_label('Classe')

plt.tight_layout()
plt.show()

### TODO: Create a 20 dimensional PCA-based feature matrix
pca= PCA(n_components=20)
F_pca = pca.fit_transform(F)

# Print reduced feature matrix shape
print(f"Feature matrix F_pca shape: {F_pca.shape}")


##########################################
## Feature engineering
##########################################
### # Function to extract zone-based features
###  Zone-Based Partitioning is a feature extraction method
### that helps break down an image into smaller meaningful regions to analyze specific patterns.
def extract_zone_features(images):
    '''Break down an 8x8 image in 3 zones: row 1-3, 4-5, and 6-8'''
    n = images.shape[0]
    mean=[]
    for i in range(n):
        img = images[i].reshape(8, 8)
        zone1 = img[0:3,:]
        zone2 = img[3:5,:]
        zone3 = img[5:8,:]
        #calcul de la moyenne des intensités de chaque zone
        mean_zone1 = np.mean(zone1)
        mean_zone2 = np.mean(zone2)
        mean_zone3 = np.mean(zone3)
        mean.append([mean_zone1, mean_zone2, mean_zone3])
    return np.array(mean)

# Apply zone-based feature extraction
F_zones = extract_zone_features(F)

# Print extracted feature shape
print(f"Feature matrix F_zones shape: {F_zones.shape}")

### Edge detection features

## TODO: Get used to the Sobel filter by applying it to an image and displaying both the original image
# and the result of applying the Sobel filter side by side
sobel_image = sobel(original_image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='RdPu')
plt.colorbar()
plt.title("Image originale")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sobel_image, cmap='RdPu')
plt.title("Avec le filtre de Sobel")
plt.colorbar()
plt.axis('off')

plt.tight_layout()
plt.show()

# TODO: Compute the average edge intensity for each image and return it as an n by 1 array
n=F.shape[0]
F_edges = np.zeros((n, 1))
for i in range(n):
    F_edges[i] = np.mean(sobel(F[i].reshape(8, 8)))

# Print feature shape after edge extraction
print(f"Feature matrix F_edges shape: {F_edges.shape}")

### connect all the features together

# TODO: Concatenate PCA, zone-based, and edge features
F_final = np.hstack([F_pca, F_zones, F_edges]) #permet de concaténer

# TODO: Normalize final features
scaler = MinMaxScaler()
F_final = scaler.fit_transform(F_final)

# Print final feature matrix shape
print(f"Final feature matrix F_final shape: {F_final.shape}")
