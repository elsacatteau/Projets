# Split the data
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multiclass import  OneVsOneClassifier,OneVsRestClassifier
from sklearn.svm import SVC
# Load processed feature matrix and labels
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from skimage.filters import sobel
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import load_digits
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


import matplotlib.pyplot as plt
import random
import time
import pandas as pd
import seaborn as sns

digits=load_digits()

def get_statistics_text(targets):
    labels, counts = np.unique(targets, return_counts=True)
    return labels, counts


def plot_multi(data, y):
    '''Plots 16 digits'''
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
    #plt.show()


# TODO: Load the raw data
X = digits.data
y = digits.target

print(X.shape[0])

#####
#In machine learning, we must train the model on one subset of data and test it on another.
#This prevents the model from memorizing the data and instead helps it generalize to unseen examples.
#The dataset is typically divided into:
#Training set → Used for model learning.
#Testing set → Used for evaluating model accuracy.
# The training set is also split as a training set and validation set for hyper-parameter tunning. This is done later
#
# Split dataset into training & testing sets


##########################################
## Train/test split and distributions
##########################################


# 1- Split dataset into training & testing sets
# TODO: FILL OUT THE CORRECT SPLITTING HERE

k=80/100
split_X = int(k * (X.shape[0])) #on prend 80% des lignes de X
split_Y = int(k * (y.shape[0])) #on prend 80% des lignes de y
X_train, X_test, y_train, y_test = X[0:split_X,:], X[split_X:,:], y[0:split_Y], y[split_Y:]


### If you want, you could save the data, this would be a good way to test your final script in the same evaluation mode as what we will be doing
# np.save("X_train.npy", X_train)
# np.save("test_data.npy", X_test)
# np.save("y_train.npy", y_train)
# np.save("test_label.npy", y_test)
####

# TODO: Print dataset split summary...
print(f"Total samples: {X.shape[0]}")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")



# TODO: ... and plot graphs of the three distributions in a readable and useful manner (bar graph, either side by side, or with some transparancy)

def plot_distribution(y_original, y_train, t_test):
    labels,counts = get_statistics_text(y_original)
    labels, counts_train = get_statistics_text(y_train)
    labels, counts_test = get_statistics_text(y_test)

    x = np.arange(len(labels))  # positions des labels sur l'axe x
    width = 0.25  # largeur des barres

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, counts, width, label='Original', color='pink')
    plt.bar(x, counts_train, width, label='Train', color='skyblue')
    plt.bar(x + width, counts_test, width, label='Test', color='lightgreen')

    plt.xlabel("Chiffre")
    plt.ylabel("Nombre d'exemples")
    plt.title("Distribution des classes dans les datasets")
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

#appel de la fonction
plot_distribution(y, y_train, y_test)


##version normalisée en divisant par le nombre de données

def plot_normalized_distribution(y_original, y_train, y_test):
    labels, counts_orig = get_statistics_text(y_original)
    _, counts_train = get_statistics_text(y_train)
    _, counts_test = get_statistics_text(y_test)

    # Normalisation par le total
    total_orig = len(y_original)
    total_train = len(y_train)
    total_test = len(y_test)

    counts_orig_norm = counts_orig / total_orig
    counts_train_norm = counts_train / total_train
    counts_test_norm = counts_test / total_test

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, counts_orig_norm, width, label='Original', color='pink')
    plt.bar(x, counts_train_norm, width, label='Train', color='skyblue')
    plt.bar(x + width, counts_test_norm, width, label='Test', color='lightgreen')

    plt.xlabel("Chiffre")
    plt.ylabel("Proportion (entre 0 et 1)")
    plt.title("Distribution des classes normalisée (par rapport au total)")
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_normalized_distribution(y, y_train, y_test)

#Remarque : sur le graphique normalisée, on observe que pour chaque classe, les 3 bâtons sont à peu près à la même hauteur ce qui nous indique que l'on a bien une bonne distribution.

# TODO: (once the learning has started, and to be documented in your report) - Impact: Changing test_size affects model training & evaluation.

##########################################
## Prepare preprocessing pipeline
##########################################

# We are trying to combine some global features fitted from the training set
# together with some hand-computed features.
#
# The PCA shall not be fitted using the test set.
# The handmade features are computed independently from the PCA
# We therefore need to concatenate the PCA computed features with the zonal and
# edge features.
# This is done with the FeatureUnion class of sklearn and then combining everything in
# a Pipeline.
#
# All elements included in the FeatureUnion and Pipeline shall have at the very least a
# .fit and .transform method.
#
# Check this documentation to understand how to work with these things
# https://scikit-learn.org/stable/auto_examples/compose/plot_feature_union.html#sphx-glr-auto-examples-compose-plot-feature-union-py

# Example of wrapper for adding a new feature to the feature matrix


##Gradient moyen dans nos trois zones : haut/milieu/bas
class EdgeInfoPreprocessing(BaseEstimator, TransformerMixin):
    '''A class used to compute an average Sobel estimator on the image
       This class can be used in conjunction of other feature engineering
       using Pipelines or FeatureUnion
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self # No fitting needed for this processing

    def transform(self, X):
        features = []
        for img in X:
            img_2d = img.reshape((8, 8))
            sobel_img = sobel(img_2d)

            mean_gradient = np.mean(sobel_img)
            features.append([mean_gradient])

        return np.array(features)

# TODO: Fill out the useful code for this class

##Intensité moyenne : haut/milieu/bas
class ZonalInfoPreprocessing(BaseEstimator, TransformerMixin):
    '''A class used to compute zone information on the image
       This class can be used in conjunction of other feature engineering
       using Pipelines or FeatureUnion

       TODO: Continue this work
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features=[]
        for img in X:
            img_2d = img.reshape((8, 8))
            top = img_2d[0:3, :]
            middle = img_2d[3:5, :]
            bottom = img_2d[5:8, :]
            means = [np.mean(top), np.mean(middle), np.mean(bottom)]
            features.append(means)
        return np.array(features)


##Création de l'objet combiné avec FeatureUnion
class PCAFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)


# TODO: Create a single sklearn object handling the computation of all features in parallel
#Création de all_features en utilisant FeatureUnion
all_features = FeatureUnion([
    ('pca', PCAFeatureExtractor(n_components=20)),
    ('zonal', ZonalInfoPreprocessing()),
    ('edge', EdgeInfoPreprocessing())
])



# Now combine everything in a Pipeline
# The clf variable is the one which plays the role of the learning algorithms
# The Pipeline simply allows to include the data preparation step into it, to
# avoid forgetting a scaling, or a feature, or ...

# TODO: Write your own pipeline, with a linear SVC classifier as the prediction

clf = Pipeline([
    ('features', all_features),
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel='linear'))
])


#Entrainement et vérification
clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))
#Accuracy correspond à la précision globale de notre pipeline clf. Ici, le modèle a correctement prédit 93.6% des chiffres sur le jeu de test.
#Cela nous indique que notre pipeline (avec PCA + features manuelles + scaling + linearSCv) est efficace.



F = all_features.fit(X_train,y_train).transform(X_train)
print("Nb features computed: ", F.shape[1]) #le nombre de features correspond à notre nombre de dimensions : 20 pour la PCA, 3 pour la zone d'intensité moyenne (haut, milieu et bas), et 1 pour le gradient moyen obtenu avec Sobel.


##########################################
## Premier entrainement d'un SVC
##########################################

# TODO: Train your model via the pipeline
clf.fit(X_train, y_train)

# TODO: Predict the outcome of the learned algorithm on the train set and then on the test set
predict_test = clf.predict(X_test)
predict_train = clf.predict(X_train)

print("Accuracy of the SVC on the test set: ", clf.score(X_test, y_test))
print("Accuracy of the SVC on the train set: ",clf.score(X_train, y_train))

# TODO: Look at confusion matrices from sklearn.metrics and
# 1. Display a print of it

conf_mat_test = confusion_matrix(y_test, predict_test)
conf_mat_train = confusion_matrix(y_train, predict_train)

print("\nConfusion matrix (test):\n", conf_mat_test)
print("\nConfusion matrix (train):\n", conf_mat_train)

# 2. Display a nice figure of it

fig1, ax1 = plt.subplots()
disp_test = ConfusionMatrixDisplay(confusion_matrix=conf_mat_test)
disp_test.plot(ax=ax1, cmap='PuRd')
ax1.set_title("Matrice de confusion - Test set")

fig2, ax2 = plt.subplots()
disp_train = ConfusionMatrixDisplay(confusion_matrix=conf_mat_train)
disp_train.plot(ax=ax2, cmap='PuRd')
ax2.set_title("Matrice de confusion - Train set")

plt.show()

# 3. Report on how you understand the results

# TODO: Work out the following questions (you may also use the score function from the classifier)
print("\n Question: How does changing test_size influence accuracy?")
print("Try different values like 0.1, 0.3, etc., and compare results.\n")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
train_scores = []
test_scores = []

for test_size in test_sizes:
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Fit
    clf.fit(X_train, y_train)

    # Accuracy
    acc_train = clf.score(X_train, y_train)
    acc_test = clf.score(X_test, y_test)

    train_scores.append(acc_train)
    test_scores.append(acc_test)

    print(f"Test size: {test_size:.1f} | Accuracy train: {acc_train:.3f} | Accuracy test: {acc_test:.3f}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(test_sizes, train_scores, label='Accuracy sur entraînement', marker='o', color='blue')
plt.plot(test_sizes, test_scores, label='Accuracy sur test', marker='s', color='orange')
plt.xlabel('Test Size (proportion)')
plt.ylabel('Accuracy')
plt.title('Impact du test_size sur la précision du modèle')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



##########################################
## Hyper parameter tuning and CV - Validation croisée
##########################################
# TODO: Change from the linear classifier to an rbf kernel
clf = Pipeline([('features', all_features),('scaler', StandardScaler()),('classifier', SVC(kernel='rbf'))])

# TODO: List all interesting parameters you may want to adapt from your preprocessing and algorithm pipeline
# TODO: Create a dictionary with all the parameters to be adapted and the ranges to be tested
param_grid = {'classifier__C': [0.01, 0.1, 1, 10, 100, 1000],'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10]}

# TODO: Use a GridSearchCV on 5 folds to optimize the hyper parameters
grid_search = GridSearchCV(estimator=clf,param_grid=param_grid,cv=5,n_jobs=1,verbose=10,scoring='accuracy',refit=True )
grid_search.fit(X_train, y_train)

# TODO: fit the grid search CV and
# 1. Check the results
print("Meilleurs paramètres trouvés: ")
print(grid_search.best_params_)
best_params = grid_search.best_params_
# On extrait les valeurs optimales
C_opt = best_params.get('classifier__C', 1)
gamma_opt = best_params.get('classifier__gamma', 'scale')

# 2. Update the original pipeline (or create a new one) with all the optimized hyper parameters
optimized_clf = Pipeline([('features', FeatureUnion([('pca', PCAFeatureExtractor(n_components=29)),
    ('sobel', EdgeInfoPreprocessing()),('zones', ZonalInfoPreprocessing())])),
    ('scaler', StandardScaler()),('classifier', SVC(kernel='rbf', C=C_opt, gamma=gamma_opt))])

# 3. Retrain on the whol train set, and evaluate on the test set
optimized_clf.fit(X_train, y_train)
test_accuracy = optimized_clf.score(X_test, y_test)
print(f"Précision sur le test set avec les meilleurs paramètres: {test_accuracy:.4f}")


# 4. Answer the questions below and report on your findings
print(" K-Fold Cross-Validation Results:")
print(f"- Best Cross-validation score: {grid_search.best_score_}")
print(f"- Best parameters found: {grid_search.best_estimator_}")


##Etude des paramètres c et gamma les plus "optimaux"
gamma_vals = [0.0001, 0.001, 0.01, 0.1, 1]
c_vals = [0.01,0.1, 1, 10, 100, 1000]

param_grid = {
    'classifier__C': c_vals,
    'classifier__gamma': gamma_vals
}

pipeline = Pipeline([
    ('features', all_features),
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel='rbf'))
])

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

results_df = pd.DataFrame(grid_search.cv_results_)

# Tableau pour la heatmap
heatmap_data = results_df.pivot_table(
    index='param_classifier__C',
    columns='param_classifier__gamma',
    values='mean_test_score'
)

#Affichage de la heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis")
plt.title("Heatmap Précision Moyenne - GridSearchCV (C vs gamma)")
plt.xlabel("gamma")
plt.ylabel("C")
plt.tight_layout()
plt.show()


##### Etude du paramètre K
print("\n Question: What happens if we change K from 5 to 10?")

#Pour K=5 :
grid_search_5 = GridSearchCV(clf, param_grid, cv=5, verbose=0, n_jobs=1, scoring='accuracy',refit=True)
grid_search_5.fit(X_train, y_train)

print(f"Best CV score with 5 folds: {grid_search_5.best_score_:.4f}")
print(f"Best parameters with 5 folds: {grid_search_5.best_params_}")


#Pour K=10 :
grid_search_10 = GridSearchCV(clf, param_grid, cv=10, verbose=0, n_jobs=1, scoring='accuracy',refit=True)
grid_search_10.fit(X_train, y_train)

print(f"Best CV score with 10 folds: {grid_search_10.best_score_:.4f}")
print(f"Best parameters with 10 folds: {grid_search_10.best_params_}")

## Etude du K "optimal"
k_values = [2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12]
cv_scores = []

# Pipeline clf et param_grid doivent être déjà définis dans votre code
for k in k_values:
    grid_search_k = GridSearchCV(clf, param_grid, cv=k, n_jobs=-1, verbose=0)
    grid_search_k.fit(X_train, y_train)
    cv_scores.append(grid_search_k.best_score_)
    print(f"K={k} | Best CV Score: {grid_search_k.best_score_:.4f}")

# Tracer le graphique
plt.figure(figsize=(8,5))
plt.plot(k_values, cv_scores, marker='o', linestyle='-', color='blue')
plt.xlabel("Nombre de plis K dans la validation croisée")
plt.ylabel("Précision moyenne (accuracy) CV")
plt.title("Variation de la précision en fonction de K")
plt.grid(True)
plt.tight_layout()
plt.show()

#On observe sur ce graphique que l'on a la meilleure précsion pour K=7

grid_search_7 = GridSearchCV(clf, param_grid, cv=7, verbose=0, n_jobs=1, scoring='accuracy',refit=True)
grid_search_7.fit(X_train, y_train)

print(f"Best CV score with 7 folds: {grid_search_7.best_score_:.4f}")
print(f"Best parameters with 7 folds: {grid_search_7.best_params_}")


##########################################
## OvO and OvR
##########################################
# TODO: Using the best found classifier, analyse the impact of one vs one versus one vs all strategies
# Analyse in terms of time performance and accuracy
best_pipeline = grid_search.best_estimator_
features = best_pipeline.named_steps["features"]
classifier = best_pipeline.named_steps["classifier"]

X_train_transformed = features.fit_transform(X_train)
X_test_transformed = features.transform(X_test)

# One-vs-One
start_ovo = time.time()
ovo = OneVsOneClassifier(classifier)
ovo.fit(X_train_transformed, y_train)
ovo_pred = ovo.predict(X_test_transformed)
ovo_time = time.time() - start_ovo
ovo_acc = accuracy_score(y_test, ovo_pred)

# Print OvO results
print(" One-vs-One (OvO) Classification:")
print("Accuracy: ",ovo_acc)
print("Time taken: ",ovo_time, "seconds")
print("Number of classifiers trained: ",len(ovo.estimators_))
print("- Impact: Suitable for small datasets but increases complexity.")

print("\n Question: How does OvO compare to OvR in execution time?")
print("Try timing both methods and analyzing efficiency.\n")

###################
# TODO:  One-vs-Rest (OvR) Classification
start_ovr = time.time()
ovr = OneVsRestClassifier(classifier)
ovr.fit(X_train_transformed, y_train)
ovr_pred = ovr.predict(X_test_transformed)
ovr_time = time.time() - start_ovr
ovr_acc = accuracy_score(y_test, ovr_pred)

# Print OvR results
print(" One-vs-Rest (OvR) Classification:")
print("Accuracy: ",ovr_acc)
print("Time taken: ",ovr_time, "seconds")
print("Number of classifiers trained: ",len(ovr.estimators_))
print("- Impact: Better for large datasets but less optimal for highly imbalanced data.")

print("\n Question: When would OvR be better than OvO?")
print("Analyze different datasets and choose the best approach!\n")
########