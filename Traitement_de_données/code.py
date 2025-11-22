# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 15:06:44 2025

@author: Camille ANSEL, ELSA CATTEAU, Anas
"""

import pandas as pd
import string
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# fonction pour retirer la ponctuation et les accents d'un texte
def remove_accents_ponctuation(text):
    accents = {
        'à': 'a', 'â': 'a', 'ä': 'a',
        'á': 'a', 'ã': 'a', 'å': 'a',
        'ç': 'c',
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'í': 'i', 'î': 'i', 'ï': 'i', 'ì': 'i',
        'ñ': 'n',
        'ó': 'o', 'ô': 'o', 'ö': 'o', 'ò': 'o', 'õ': 'o',
        'ù': 'u', 'û': 'u', 'ü': 'u', 'ú': 'u',
        'ý': 'y', 'ÿ': 'y',
        ',': ' ', ';': ' ', ':': ' ', '?': ' ', '.': ' ', '/': ' ', '!': ' ', '_': ' ',
        '"': ' ', '-': ' ', '&': ' ', '(': ' ', ')': ' ', '[': ' ', ']': ' '
    }
    
    return ''.join(accents.get(char, char) for char in text)

# liste de mots vides
stop_words = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'with', 'to', 'for', 'on',
    'in', 'at', 'by', 'of', 'up', 'out', 'as', 'is', 'it', 'this', 'that', 'these',
    'those', 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'my', 'your', 'his', 'her',
    'their', 'our', 'us', 'from',
    'la','le',
    'op', 'feat', 'remastered','live','remix','version','major','minor','allegro',
    '1','2','3','4','5',
    'ii','iii'
}


def has_no(title):
    """
    Détecte si un titre contient une expression de négation
    dans l'une des langues suivantes : anglais, français, espagnol, allemand, 
    arabe, russe, chinois (mandarin), japonais, coréen.

    La fonction vérifie à la fois les mots séparés (ex: "no", "non", "nein")
    et les caractères liés à la négation dans des langues non séparées par espaces 
    (ex: chinois, japonais, coréen).

    Paramètres
    ----------
    title : str
        Le titre de la chanson à analyser (doit être déjà en minuscules, sans accents).

    Retour
    ------
    int
        1 si au moins un mot ou caractère de négation est détecté, 0 sinon.
    """
    if not isinstance(title, str):
        return 0

    # Liste multilingue de mots ou caractères exprimant la négation
    negation_words = {
        'no', 'not',                 # anglais
        'non', 'pas',               # français
        'nein', 'nicht', 'kein',    # allemand
        'не', 'нет', 'ни', 'без', 'вовсе', # russe
        'لا'        
        }
    negation_symbols = {
        'ない', 'じゃない',             # japonais
        '不', '没',                 # mandarin
        '아니', '없다'                # coréen
    }

    tokens = title.split()  # découpe simple en mots (pour langues à espaces)

    for word in negation_words:
        # Vérifie la présence soit comme mot isolé
        if word in tokens:
            return 1
    for symbol in negation_symbols:
        # soit comme séquence dans la chaîne
        if symbol in title:
            return 1
    return 0

def has_love(title):
    """
    Détecte si un titre contient une expression d'amour ("love")
    dans l'une des langues suivantes : anglais, français, espagnol, 
    allemand, arabe, russe, chinois, japonais, coréen.

    La détection se fait par recherche de mots complets (pour langues à espaces)
    ou de séquences de caractères (pour langues comme le chinois ou le japonais).

    Paramètres
    ----------
    title : str
        Le titre de la chanson à analyser (doit être déjà en minuscules, sans accents si nécessaire).

    Retour
    ------
    int
        1 si au moins un mot ou caractère lié à "amour" est détecté, 0 sinon.
    """
    if not isinstance(title, str):
        return 0

    love_words = {
        # Anglais
        'love',
        # Français
        'amour', 'aimer', 'aime',
        # Espagnol
        'amor', 'querer', 'amar',
        # Allemand
        'liebe', 'lieben',
        # Arabe
        'حب', 'حبيب', 'احبك', 'عشق', 'غرام',
        # Russe
        'любовь', 'люблю', 'любить', 'милая',
    }
    love_symbols = {
        # Chinois (Mandarin)
        '爱', '爱情', '爱你',
        # Japonais
        '愛', '好き', '愛してる',
        # Coréen
        '사랑', '사랑해', '좋아해'
    }

    tokens = title.split()

    for word in love_words:
        if word in tokens:
            return 1
    for symbol in love_symbols:
        if symbol in title:
            return 1
    return 0


def has_top_word(title):
    """
    Permet de savoir si un titre contient un des mots les plus utilisés

    Parameters
    ----------
    title : string
        le titre du morceau

    Returns
    -------
    boolean
        true si le titre contient un des mots les plus fréquents dans les titres du dataset.

    """
    if not isinstance(title, str):
        return 0
    return int(any(word in top_words for word in title.split()))


# Récupération des données
df = pd.read_csv('music_genre_1.csv') # On passe au format dataFrame
print(df.head())
## Nettoyage
# On supprime les lignes vides
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.isnull().sum()) #on voit qu'il n'y a plus de lignes vides

# Convertir les noms de notes en one-hot
key_dummies = pd.get_dummies(df['key'], prefix='key')
df=pd.concat([df, key_dummies], axis=1)
genre_dummies = pd.get_dummies(df['music_genre'], prefix='music_genre')
df=pd.concat([df, genre_dummies], axis=1)
# Conversion en secondes des durées
df['duration_sec'] = df['duration_ms'] / 1000 
# On supprime les lignes pour lesquelles la durée n'est pas renseignée
nb_musiques_sans_duree = len(df[df['duration_ms'] < 0])
print(f"Nombre de musiques dont la durée n'est pas renseignée' : {nb_musiques_sans_duree}")
df = df[df['duration_ms'] != -1]
# On observe qu'une minorité de musique ont une durée très grande
nb_musiques_dix_mins = len(df[df['duration_sec'] > 600])
print(f"Nombre de musiques dont la durée dépasse 10 minutes' : {nb_musiques_dix_mins}")

# On remplace les ? par NaN
df['tempo'] = df['tempo'].replace('?', pd.NA)
df['tempo'] = pd.to_numeric(df['tempo'])
#On enlève les lignes où il y a NaN
df = df.dropna(subset=['tempo'])

# On enlève les éventuels doublons
df = df.drop_duplicates()

# On enlève les colonnes inutiles
df = df.drop('obtained_date', axis=1)
df = df.drop('instance_id', axis=1)
df = df.drop('duration_ms', axis=1)

# Mise en forme des titres
df['title_clean'] = df['track_name'].str.lower().apply(remove_accents_ponctuation)
# On souhaite trouver les idées qui reviennent souvent dans les titres
# On extrait les mots de titres
all_words = ' '.join(df['title_clean']).split() # Combiner tous les titres en une seule liste de mots
filtered_words = [word for word in all_words if len(word) > 1 and word not in stop_words] # Supprimer les stopwords
word_counts = Counter(filtered_words) # Compter les mots
print(word_counts.most_common(100)) # Voir les 100 mots les plus fréquents
# Le titre contient-il ces mots très répandus ?
df['has_negation'] = df['title_clean'].apply(has_no)
df['has_love'] = df['title_clean'].apply(has_love)
# Extraire les mots les plus fréquents
top = 100
top_words = set([word for word, _ in word_counts.most_common(top)])
df['has_top'] = df['title_clean'].apply(has_top_word)

#  premières observations
prop_negation = df['has_negation'].mean()
prop_love = df['has_love'].mean()
prop_top = df['has_top'].mean()
print(f"Proportion de titres contenant une négation : {prop_negation:.2%}")
print(f"Proportion de titres contenant une idée d'amour : {prop_love:.2%}")
print(f"Proportion de titres contenant un mot appartenant au top {top} : {prop_top:.2%}")
"""
#Graphe genre musical en fonction de la popularité
plt.figure(figsize=(12, 6))
sns.boxplot(x='music_genre', y='popularity',data=df)
sns.pointplot(x='music_genre', y='popularity', data=df, estimator='mean', color='red', linestyles='', markers='o')
# Ajout du titre et des légendes
plt.title("Graphe de la popularité selon le genre musical")
plt.xlabel("Genre musical")
plt.ylabel("Popularité (en %)")
# Affichage
plt.tight_layout()
plt.show()

#Graphe has_love en fonction de la popularité
plt.figure(figsize=(12, 6))
sns.barplot(x='has_love', y='popularity', data=df)
plt.title("Popularité moyenne selon la présence d'un mot signifiant amour dans le titre")
plt.xlabel("Titre contient un mot signifiant amour (0 = non, 1 = oui)")
plt.ylabel("Popularité moyenne (en %)")
plt.show()

#Graphe has_negation en fonction de la popularité
plt.figure(figsize=(12, 6))
sns.barplot(x='has_negation', y='popularity', data=df)
plt.title("Popularité moyenne selon la présence d'un mot négatif dans le titre")
plt.xlabel("Titre contient un mot négatif (0 = non, 1 = oui)")
plt.ylabel("Popularité moyenne (en %)")
plt.show()

#Graphe has_top en fonction de la popularité
plt.figure(figsize=(12, 6))
sns.barplot(x='has_top', y='popularity', data=df)
plt.title("Popularité moyenne selon la présence d'un mot dans le top 50 dans le titre")
plt.xlabel("Titre contient un mot dans le top 50 (0 = non, 1 = oui)")
plt.ylabel("Popularité moyenne (en %)")
plt.show()

nb_artistes = df['artist_name'].nunique()
print(f"Nombre d'artistes différents : {nb_artistes}")
nb_musiques_par_artiste = df['artist_name'].value_counts()
print(nb_musiques_par_artiste)

#Graphe popularité des artistes en fonction du nombres d'artistes
# Calculer la popularité moyenne par artiste
mean_pop_by_artist = df.groupby('artist_name')['popularity'].mean()
print(mean_pop_by_artist)
# Tracer un histogramme des popularités moyennes
plt.figure(figsize=(12, 6))
sns.histplot(mean_pop_by_artist, bins=15, kde=False, color='skyblue')
plt.title("Distribution des artistes selon leur popularité moyenne")
plt.xlabel("Popularité moyenne de l'artiste")
plt.ylabel("Nombre d'artistes")
plt.tight_layout()
plt.show()


#Graphe de la danceability en fonction de loudness
# Nombre de morceaux à conserver par genre
n_par_genre = 200
# Sous-échantillonnage stratifié
df_sampled = df.groupby('music_genre').apply(lambda x: x.sample(min(len(x), n_par_genre), random_state=42))
df_sampled = df_sampled.reset_index(drop=True)
#Graphe
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_sampled, x='loudness', y='danceability', hue='music_genre', alpha=0.6)
plt.title("Danceability en fonction du Loudness")
plt.xlabel("Loudness")
plt.ylabel("Danceability")
plt.grid(True)
plt.show()"""

#Corrélation
df['acoustic_dance']=-(df['acousticness']*df['danceability'])
df['acoustic_energy']=-(df['acousticness']*df['energy'])
df['acoustic_instrument']=df['acousticness']*df['instrumentalness']
df['acoustic_loudness']=-(df['acousticness']*df['loudness'])
df['acoustic_valence']=-(df['acousticness']*df['valence'])
df['dance_energy']=df['danceability']*df['energy']
df['dance_instrument']=-(df['danceability']*df['instrumentalness'])
df['dance_loudness']=df['danceability']*df['loudness']
df['dance_speech']=df['danceability']*df['speechiness']
df['dance_valence']=df['danceability']*df['valence']
df['energy_instrument']=-(df['energy']*df['instrumentalness'])
df['energy_loudness']=df['energy']*df['loudness']
df['energy_valence']=df['energy']*df['valence']
df['energy_hasno']=df['energy']*df['has_negation']
df['instrumental_loudness'] = -(df['instrumentalness'] * df['loudness'])
df['instrumental_valence'] = -(df['instrumentalness'] * df['valence']) 
df['loudness_hasno'] = df['loudness'] * df['has_negation'] 
df['loudness_valence'] = df['loudness'] * df['valence'] 

key_columns = [col for col in df.columns if col.startswith('key_')]
genre_columns = [col for col in df.columns if col.startswith('music_genre_')]

# Variables explicatives (features)
features = [
    'danceability', 'energy', 'loudness', 'liveness','tempo','speechiness', 'acousticness','instrumentalness', 'valence',
    'duration_sec', 'has_negation', 'has_love','has_top','acoustic_dance','acoustic_energy','acoustic_valence',
    'acoustic_loudness', 'dance_energy','dance_loudness','dance_valence','energy_loudness', 'energy_valence','loudness_valence',
    'acoustic_instrument',  'dance_instrument', 'dance_speech', 'energy_instrument', 'energy_hasno', 'instrumental_loudness','instrumental_valence',
    'loudness_hasno',] + key_columns + genre_columns

X = df[features]
y = df['popularity']
# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
#R carré ajusté
n = X_test.shape[0]  # nombre d'observations dans le test
p = X_test.shape[1]  # nombre de variables explicatives
r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
print(f"R² ajusté : {r2_adj:.2f}")
print(f"R² Score : {r2:.2f}")

#Test comparaison de la popularité prédite avec la vraie popularité pour une seule musique
i = 8
musique = X.iloc[[i]]
y_true = y.iloc[i]
y_pred = model.predict(musique)[0]

print("Titre :", df.iloc[i]['track_name'])
print("Artiste :", df.iloc[i]['artist_name'])
print(f"Popularité réelle : {y_true}")
print(f"Popularité prédite : {y_pred:.2f}")
# Affichage des coefficients
coefficients = pd.Series(model.coef_, index=features)
print("Coefficients du modèle :")
print(coefficients.sort_values(ascending=False))

#Régression logistique
# Variable cible binaire : populaire ou non
# Seuil de popularité à définir (par exemple, 50)
df['is_popular'] = (df['popularity'] >= 50).astype(int)
# Cible binaire
y_logistic = df['is_popular']

# Même features que pour la régression linéaire
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_logistic, test_size=0.3, random_state=42)

# Régression logistique
log_model = LogisticRegression(max_iter=10000)
log_model.fit(X_train_log, y_train_log)

# Prédictions
y_pred_log = log_model.predict(X_test_log)

# Évaluation
print(confusion_matrix(y_test_log, y_pred_log))
print(classification_report(y_test_log, y_pred_log))
ConfusionMatrixDisplay.from_estimator(log_model, X_test_log, y_test_log)
plt.title('Matrice de Confusion (Régression Logistique)')
plt.show()

