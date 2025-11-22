import pandas as pd
import folium #librairie pour avoir une carte intéractive
from datetime import datetime, timedelta

# Charger le fichier
df = pd.read_csv("tgv_sans_emission.csv", sep=";")

# Création d'une carte centrée sur la France
m = folium.Map(location=[47, 2.5], zoom_start=6)

def calculer_duree(depart, arrivee):
    format_h= "%H:%M" #format de l'heure
    t_depart = datetime.strptime(depart, format_h)  #convertion en datatime
    t_arrivee = datetime.strptime(arrivee, format_h)

    duree = t_arrivee - t_depart
    heures, reste = divmod(duree.seconds, 3600) #division par 3600
    minutes = reste // 60
    return f"{heures}h {minutes}min"

def afficher_trajets(gare_nom):
    trajets = df[df['Origine'].str.upper() == gare_nom.upper()] #garde la colonne origine et compare avec gare_nom (en majuscule)

    if trajets.empty: #si aucun trajet trouvé
        print(f"Aucun trajet trouvé depuis {gare_nom}") 
        return

    # Marqueur pour la gare d'origine
    lat, lon = trajets.iloc[0]['latitude_origine'], trajets.iloc[0]['longitude_origine']
    folium.Marker([lat, lon], popup=f"<b>Départ :</b> {gare_nom}",
                  icon=folium.Icon(color="red", icon="train")).add_to(m) # marquer la gare d'origine

    # Ajouter toutes les destinations
    for col , row in trajets.iterrows():
        lat_dest, lon_dest = row['latitude_destination'], row['longitude_destination']
        duree_str = calculer_duree(row['Heure_depart'], row['Heure_arrivee']) # calcul de la durée du trajet
        popup_txt = f"""Destination : {row['Destination']}<br>
                    <b>Durée du trajet :</b> {duree_str}""" # ce qui s'affiche en cliquant sur la destination
        
        # Marquer la gare de destination par un point
        folium.Marker([lat_dest, lon_dest], popup=folium.Popup(popup_txt, max_width=300),icon=folium.Icon(color="blue", icon="flag")).add_to(m)

        # Ligne entre origine et destination
        folium.PolyLine([[lat, lon], [lat_dest, lon_dest]],color="green", weight=2).add_to(m)

# Demander la gare à l'utilisateur
gare_depart = input("Entrez le nom de la gare de départ : ")

# Générer la carte
afficher_trajets(gare_depart)

# Sauvegarder la carte en html
m.save("carte_sans_emission.html")
print("Carte générée : carte_sans_emission.html")