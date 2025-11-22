import pandas as pd
import folium

# Charger le fichier
df = pd.read_csv("tgv_avec_emission.csv", sep=";")

# Création d'une carte centrée sur la France
m = folium.Map(location=[47, 2.5], zoom_start=6) # centre de la carte placé à la latitude 47° Nord et longitude 2.5° Est

def afficher_trajets(gare_nom):
    trajets = df[df['Origine'].str.upper() == gare_nom.upper()]

    if trajets.empty:
        print(f"Aucun trajet trouvé depuis {gare_nom}")
        return

    # Marqueur pour la gare d'origine
    lat, lon = trajets.iloc[0]['latitude_origine'], trajets.iloc[0]['longitude_origine']
    folium.Marker([lat, lon], popup=f"<b>Départ :</b> {gare_nom}",icon=folium.Icon(color="red", icon="train")).add_to(m)

    # Ajouter toutes les destinations
    for index, row in trajets.iterrows():
        lat_dest, lon_dest = row['latitude_destination'], row['longitude_destination']
        
        popup_txt = popup_html = f"""
        <b>Destination :</b> {row['Destination']}<br>
        <b>Distance :</b> {row['Distance entre les gares']} km<br><br>
        <b>Émissions (kgCO2e) :</b><br>
        Train : {row['Train - Empreinte carbone (kgCO2e)']}<br>
        Autocar : {row['Autocar longue distance - Empreinte carbone (kgCO2e)']}<br>
        Avion : {row['Avion - Empreinte carbone (kgCO2e)']}<br>
        Voiture : {row['Voiture thermique (2,2 pers.) - Empreinte carbone (kgCO2e)']}
        """
        
        # Marquer la destination par un point
        folium.Marker([lat_dest, lon_dest], popup=folium.Popup(popup_txt, max_width=300),icon=folium.Icon(color="blue", icon="flag")).add_to(m)

        # Ligne entre origine et destination
        folium.PolyLine([[lat, lon], [lat_dest, lon_dest]],color="green", weight=2).add_to(m)

# Demander la gare à l'utilisateur
gare_depart = input("Entrez le nom de la gare de départ : ")

# Générer la carte
afficher_trajets(gare_depart)

# Sauvegarder la carte
m.save("carte_emission.html")
print("Carte générée : carte_emission.html")
