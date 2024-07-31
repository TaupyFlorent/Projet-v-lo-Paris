import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io


@st.cache_data
def lire_csv(fichier_csv, sep):
    df = pd.read_csv(fichier_csv, sep=sep)
    return df

df = lire_csv('comptage-velo-donnees-compteurs.csv', sep=';')


st.title("Comptage cycliste Paris")
st.sidebar.title("Sommaire")
pages=["Le Projet", "Jeux de données", "DataVizualisation", 'Machine Learning', 'Conclusion']
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0]:
    st.write("## 🚴‍♂️ Le Projet 🚴‍♀️")
    
    st.write("""
    Depuis plusieurs années, la **Ville de Paris** a entrepris le déploiement de **compteurs à vélo permanents** sur ses pistes cyclables. Ces dispositifs enregistrent le nombre de cyclistes passant par heure et par piste, fournissant des **données essentielles** pour évaluer l'utilisation du réseau cyclable parisien.
    
    ### Objectif Principal
    L'objectif principal de ce projet est d'obtenir une vision précise de la **pratique du cyclisme à Paris**, permettant ainsi de guider les investissements financiers et d'aménager efficacement l'infrastructure urbaine.
    
    ### Importance du Projet
    Promouvoir des **modes de déplacement écologiques** est une priorité pour la mairie de Paris. En conséquence, ce projet, en cours depuis plusieurs années, revêt une importance capitale pour le **développement durable** de la ville.
    
    ### Utilisation des Données
    Les données recueillies sont cruciales pour orienter les décisions **politiques et financières**, visant à améliorer la **qualité de vie des Parisiens** tout en soutenant la **transition vers des modes de transport plus respectueux de l'environnement**.
    """)
    
    st.markdown("""
    **Points Clés du Projet :**
    - 🌿 **Développement Durable** : Soutenir la transition écologique.
    - 🚲 **Infrastructure Cyclable** : Améliorer et étendre les pistes cyclables.
    - 📊 **Données Précieuses** : Informer les décisions politiques et financières.
    - 🏙️ **Qualité de Vie** : Améliorer la vie quotidienne des Parisiens.
    """)

    st.write("### 🗺️ Impact sur la Ville")
    st.write("Ce projet permet de mieux comprendre et d'analyser les flux de cyclistes, facilitant ainsi la planification urbaine et les investissements futurs dans les infrastructures de transport.")


if page == pages[1]:
    st.write("## 📊 Jeux de Données")

    st.write("""
    L'objectif de cette section est de vous présenter le **jeu de données de base** sur lequel nous avons travaillé. Nous allons :
    - **Corriger** et **optimiser** ce dataset.
    - Introduire les **données météorologiques** que nous avons ajoutées pour approfondir l'analyse.

    Ces étapes sont cruciales pour garantir la précision et la pertinence de nos conclusions.
    """)

    st.write("""
    La **Ville de Paris** utilise des **compteurs vélo permanents** pour évaluer l'usage des pistes cyclables. Selon l'aménagement, un site de comptage peut être équipé de :
    - **Un compteur** pour les pistes cyclables unidirectionnelles.
    - **Deux compteurs** pour les pistes cyclables bidirectionnelles.

    ### Description du Jeu de Données
    Ce jeu de données contient les comptages vélo horaires sur une période glissante de **13 mois** (J-13 mois), mis à jour quotidiennement (J-1). Les compteurs sont installés :
    - Sur des **pistes cyclables**.
    - Dans certains **couloirs bus ouverts aux vélos**.

    **Remarque** : Les autres véhicules (ex. : trottinettes) ne sont pas comptés.

    ### Source des Données
    Les données sont fournies quotidiennement via l'API de notre partenaire **Eco Compteur**. Comme l'API ne fournit pas nativement le comptage par sens, un traitement par agrégation a été effectué pour le reconstituer.

    ### Détails Inclus
    - **Identifiant du compteur**
    - **Nom du compteur**
    - **Identifiant du site de comptage**
    - **Nom du site de comptage**
    - **Comptage horaire**
    - **Date et heure de comptage**
    - **Date d'installation du site de comptage**
    - **Lien vers photo du site de comptage**
    - **Coordonnées géographiques**
    """)

    afficher_dataset = st.checkbox('Afficher le dataset')

    # Si la case est cochée, afficher le dataset
    if afficher_dataset:
        st.write("### Dataset")
        st.dataframe(df.head())

    st.write("### Info Dataset")
    st.write("Voici les infos sur le dataset de base :")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    # Display the captured info
    st.text(info_str)

    st.write("### Correction du Dataset")
    st.write("Après analyse du Dataset, je décide de supprimer les colonnes suivantes qui me paraissent inutiles :")
    st.code("""
    df = df.drop(columns=[
    'Identifiant du site de comptage', 
    'Nom du site de comptage', 
    'Date d\'installation du site de comptage',
    'Lien vers photo du site de comptage', 
    'Identifiant technique compteur',
    'ID Photos', 
    'test_lien_vers_photos_du_site_de_comptage_',
    'id_photo_1', 
    'url_sites', 
    'type_dimage', 
    'mois_annee_comptage'
    ], axis=1)
    """)

    df2 = lire_csv('df3_paris_velo.csv', sep=',')

    st.write("Je décide ensuite de renommer certaines colonnes, de séparer certaines colonnes en deux pour faciliter ensuite mon analyse et pouvoir gagner en précision")
    st.write("Je rajoute ensuite des données météo pour pouvoir affiner mon analyse. Voici le DataSet df_meteo :")

    df_meteo = lire_csv('df_meteo.csv', sep=',')
    df_merged_meteo = lire_csv('df_merged_meteo.csv', sep=',')

    st.write("""
    #### Renommage et Transformation des Colonnes

    Pour faciliter notre analyse et gagner en précision, nous avons renommé certaines colonnes et séparé certaines d'entre elles en deux. Cela nous permettra d'examiner plus en détail les différentes facettes des données.

    #### Ajout des Données Météo

    Afin d'affiner notre analyse, nous avons intégré des données météorologiques. Voici un aperçu du DataFrame contenant ces informations :
    """)

    st.write(df_meteo.head())

    st.write("""
    **Description des colonnes :**
    - **RR1** : Quantité de pluie tombée (en mm).
    - **T** : Température relevée heure par heure (en °C).

    Ces données météorologiques nous permettront de mieux comprendre l'impact des conditions climatiques sur la pratique du cyclisme à Paris.
    """)

    st.write("""
    ### DataSet Final

    Après avoir fusionné et modifié les données, nous obtenons le dataset final qui sera utilisé pour le projet de machine learning. Voici un aperçu de ce dataset :
    """)

    st.write(df_merged_meteo.head())

    st.write("""
    **Remarque** : Les colonnes relatives au mois, au jour et à l'heure ont été transformées à l'aide de fonctions sinus/cosinus pour capturer leur caractère cyclique, améliorant ainsi l'efficacité de nos modèles de machine learning.
    """)

if page == pages[2]:
    st.write("## 🚴‍♂️ Data Visualization 🚴‍♀️")

    st.write("#### Analyse géographique")

    st.write("""
    L'objectif de cette section est de présenter divers graphiques illustrant l'impact de différentes données sur le comptage des cyclistes à Paris.
    Nous allons commencer par analyser la répartition du traffic d'un point de vue géographique.
    """)

    df3 = lire_csv('df3.csv', sep=',')

    import pydeck as pdk

    # Fusionner les données en sommant les valeurs de comptage pour chaque latitude et longitude
    df_aggregated = df3.groupby(['Latitude', 'Longitude'], as_index=False).agg({'Comptage': 'sum'})

    # Normaliser les tailles de cercles
    max_comptage = df_aggregated['Comptage'].max()
    min_comptage = df_aggregated['Comptage'].min()

    # Pour éviter des cercles de taille nulle, ajouter un petit minimum
    df_aggregated['Normalized_Comptage'] = df_aggregated['Comptage'].apply(lambda x: (x - min_comptage) / (max_comptage - min_comptage) * 1000 + 10)

    # Définir le centre de la carte
    map_center = [48.8566, 2.3522]  # Centre de Paris

    # Créer la couche de cercles
    layer = pdk.Layer(
        "ScatterplotLayer",
        df_aggregated,
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_scale=1,  # Ajustez ce facteur pour changer l'échelle des cercles
        radius_min_pixels=5,  # Taille minimale des cercles en pixels
        radius_max_pixels=100,  # Taille maximale des cercles en pixels
        line_width_min_pixels=1,
        get_position='[Longitude, Latitude]',
        get_radius='Normalized_Comptage',  # Utilisez les valeurs normalisées pour définir le rayon
        get_fill_color='[255, 0, 0, 140]',  # Couleur des cercles
        get_line_color=[0, 0, 0],  # Couleur des bords des cercles
    )

    # Créer la vue de la carte
    view_state = pdk.ViewState(
        longitude=map_center[1],
        latitude=map_center[0],
        zoom=12,  # Ajustez le niveau de zoom pour Paris
        pitch=0,
    )

    # Rendre la carte
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{Comptage} cyclistes"})

    # Afficher la carte dans Streamlit
    st.pydeck_chart(r)

    st.write("""
    On observe une concentration plus élevée de cyclistes le long des boulevards périphériques dans le sud de Paris ainsi que dans le centre-ville, particulièrement près des quais de la Seine.""")

    st.write("#### Analyse par mois")

    st.write("""
    Nous allons maintenant analyser le trafic vélo en fonction des mois.
    """)

    import matplotlib.pyplot as plt
    import calendar

    # Lecture des fichiers CSV
    df_meteo = lire_csv('df_meteo.csv', sep=',')
    df_merged_meteo = lire_csv('df_merged_meteo.csv', sep=',')

    # S'assurer que la colonne datetime est au format datetime
    df_merged_meteo['datetime'] = pd.to_datetime(df_merged_meteo['datetime'])

    # Grouper les données par mois et calculer le total de cyclistes pour chaque mois
    df_merged_meteo['Mois'] = df_merged_meteo['datetime'].dt.to_period('M')
    monthly_counts = df_merged_meteo.groupby('Mois')['Comptage'].sum().reset_index()

    # Obtenir les noms des mois
    monthly_counts['Month_Name'] = monthly_counts['Mois'].dt.month.apply(lambda x: calendar.month_name[x])

    # Créer le graphique en barres
    fig, ax = plt.subplots(figsize=(24, 12))  # Agrandir encore plus la taille de la figure
    ax.bar(monthly_counts['Month_Name'], monthly_counts['Comptage'], color='skyblue')
    ax.set_xlabel('Mois', fontsize=20)  # Augmenter la taille de la police du label x
    ax.set_ylabel('Nombre', fontsize=20)  # Augmenter la taille de la police du label y
    ax.set_title('Nombre Total de Cyclistes par Mois', fontsize=24)  # Augmenter la taille de la police du titre

    # Ajuster la taille des étiquettes des axes x et y
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Afficher le graphique dans Streamlit
    st.pyplot(fig)

    st.write("""
    Nous nous apercevons que les mois ont un impact significatif sur le trafic cyclable à Paris. 
    Il apparaît que les mois ensoleillés (hors grandes vacances scolaires) sont plus propices à l'utilisation du vélo.
    """)

    st.write("#### Analyse par jour")

    st.write("""
    Nous allons maintenant analyser le trafic vélo en fonction du jour de la semaine.
    """)

    # Assurez-vous que la colonne de date/heure est au format datetime
    df_merged_meteo['datetime'] = pd.to_datetime(df_merged_meteo['datetime'])

    # Extraire le jour de la semaine de la colonne datetime (0 = lundi, 6 = dimanche)
    df_merged_meteo['Jour'] = df_merged_meteo['datetime'].dt.dayofweek

    # Grouper les données par jour et calculer le total de cyclistes pour chaque jour
    daily_counts = df_merged_meteo.groupby('Jour')['Comptage'].sum().reset_index()

    # Obtenez les noms des jours de la semaine
    daily_counts['Day_Name'] = daily_counts['Jour'].apply(lambda x: calendar.day_name[x])

    # Créez le graphique en barres avec des ajustements pour les titres et labels
    fig, ax = plt.subplots(figsize=(24, 12))  # Agrandir encore plus la taille de la figure
    ax.bar(daily_counts['Day_Name'], daily_counts['Comptage'], color='skyblue')
    ax.set_xlabel('Jour de la semaine', fontsize=20)  # Augmenter la taille de la police du label x
    ax.set_ylabel('Nombre', fontsize=20)  # Augmenter la taille de la police du label y
    ax.set_title('Nombre Total de Cyclistes par Jour de la Semaine', fontsize=24)  # Augmenter la taille de la police du titre

    # Ajuster la taille des étiquettes des axes x et y
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)

    st.write("")
    st.write("")

    st.write("Nous observons que le nombre de cyclistes varie en fonction du jour de la semaine. Les jours de semaine montrent généralement une augmentation du nombre de cyclistes par rapport aux week-ends.")

    st.write("#### Analyse par heure")

    st.write("""
    Nous allons maintenant analyser le trafic vélo en fonction de l'heure.
    """)

    # Extraire l'heure de la colonne datetime
    df_merged_meteo['Heure'] = df_merged_meteo['datetime'].dt.hour

    # Grouper les données par heure et calculer le total de cyclistes pour chaque heure
    hourly_counts = df_merged_meteo.groupby('Heure')['Comptage'].sum().reset_index()

    # Créez le graphique en barres avec des ajustements pour les titres et labels
    fig, ax = plt.subplots(figsize=(24, 12))  # Agrandir encore plus la taille de la figure
    ax.bar(hourly_counts['Heure'], hourly_counts['Comptage'], color='skyblue')
    ax.set_xlabel('Heure', fontsize=20)  # Augmenter la taille de la police du label x
    ax.set_ylabel('Nombre', fontsize=20)  # Augmenter la taille de la police du label y
    ax.set_title('Nombre Total de Cyclistes par Heure', fontsize=24)  # Augmenter la taille de la police du titre

    # Ajuster la taille des étiquettes des axes x et y
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    plt.xticks(range(0, 24))
    plt.tight_layout()

    st.pyplot(fig)

    st.write("")
    st.write("")

    st.write("Nous observons que le nombre de cyclistes varie en fonction de l'heure de la journée. Les heures de pointe, généralement le matin et en fin de journée, montrent une augmentation significative du nombre de cyclistes.")

    st.write("### Influence de la Quantité de Pluie sur le Trafic Cycliste")

    st.write("""
    Nous allons maintenant analyser l'influence de la quantité de pluie tombée sur le trafic cycliste. Nous filtrons les données pour n'inclure que les heures où la quantité de pluie est inférieur à 15mm.
    """)

    # Filtrer les données pour n'inclure que les valeurs où la quantité de pluie (RR1) est supérieure à 15
    df_filtered = df_merged_meteo[df_merged_meteo['RR1'] < 15]

    # Créez le graphique en nuage de points pour visualiser la relation entre la quantité de pluie et le nombre de cyclistes
    fig, ax = plt.subplots(figsize=(24, 12))  # Taille de la figure
    scatter = ax.scatter(df_filtered['RR1'], df_filtered['Comptage'], alpha=0.5, color='skyblue')
    ax.set_xlabel('Quantité de Pluie (RR1)', fontsize=20)  # Augmenter la taille de la police du label x
    ax.set_ylabel('Nombre de Cyclistes', fontsize=20)  # Augmenter la taille de la police du label y
    ax.set_title('Influence de la Quantité de Pluie sur le Trafic Cycliste', fontsize=24)  # Augmenter la taille de la police du titre

    # Ajuster la taille des étiquettes des axes x et y
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    plt.tight_layout()

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)

    st.write("")
    st.write("")

    st.write("""
    Nous observons que plus la quantité de pluie augmente, plus il y a une diminution significative du nombre de cyclistes. Cette analyse nous permet de comprendre l'impact important des fortes pluies sur le trafic cycliste.
    """)

    st.write("### Influence de la Température sur le Trafic Cycliste")

    st.write("""
    Nous allons maintenant analyser l'influence de la température sur le trafic cycliste.
    """)

    # Filtrer les données pour n'inclure que les valeurs où la température est valide (par exemple, supérieures à une valeur minimale si nécessaire)
    df_filtered_temp = df_merged_meteo[df_merged_meteo['T'].notnull()]

    # Créez le graphique en nuage de points pour visualiser la relation entre la température et le nombre de cyclistes
    fig, ax = plt.subplots(figsize=(24, 12))  # Taille de la figure
    scatter = ax.scatter(df_filtered_temp['T'], df_filtered_temp['Comptage'], alpha=0.5, color='skyblue')
    ax.set_xlabel('Température (°C)', fontsize=20)  # Augmenter la taille de la police du label x
    ax.set_ylabel('Nombre de Cyclistes', fontsize=20)  # Augmenter la taille de la police du label y
    ax.set_title('Influence de la Température sur le Trafic Cycliste', fontsize=24)  # Augmenter la taille de la police du titre

    # Ajuster la taille des étiquettes des axes x et y
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    plt.tight_layout()

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)

    st.write("")
    st.write("")

    st.write("""
    Nous observons que la température a un impact significatif sur le nombre de cyclistes. En général, plus la température va dans les extrêmes (trop froid ou trop chaud) plus le nombre de cyliste baisse.
    """)

if page == pages[3]:
    st.write("## 🚴‍♂️ Machine Learning 🚴‍♀️")

    st.write("#### Les différents modèles utilisés")

    st.write("""
    Nous allons à présent procéder à des expérimentations de Machine Learning : arriver anticiper le comptage horaire futur à partir des données passées. Nous devrons ici utiliser des méthodes de Régression puisque nous avons uniquement des variables quantitatives à notre disposition.
    Pour mener à bien notre résolution de problème de Machine Learning, nous allons essayer les modèles suivants :
             
             
    -	Le Decision Tree Regressor
    -	Le XGBoost
    -	Le Random Forest Regressor
    -	Le Gradient Boosting Regressor
             
    Nous allons ensuite, comparer les performances de ces différents modèles afin de séléctionner le meilleur.

    """)

    st.write("#### 1/ Le Decision Tree Regressor")

    st.write("###### Résultats :")
    st.write("""
    - **Score du jeu d'entraînement du modèle utilisant le DecisionTreeRegressor** : 1.0
    - **Score du jeu de test du modèle utilisant le DecisionTreeRegressor** : 0.604
    - **Erreur quadratique moyenne (MSE) sur les données d'entraînement avec DecisionTreeRegressor** : 0.0
    - **Erreur quadratique moyenne (MSE) sur les données de test avec DecisionTreeRegressor** : 1387.84
    - **Erreur absolue moyenne (MAE) sur les données d'entraînement avec DecisionTreeRegressor** : 0.0
    - **Erreur absolue moyenne (MAE) sur les données de test avec DecisionTreeRegressor** : 18.30
    """)

    st.write("""
    ###### Interprétation :
    Les résultats obtenus avec le modèle DecisionTreeRegressor montrent une performance parfaite sur les données d'entraînement, avec un score de 1.0 et une erreur quadratique moyenne (MSE) de 0.0, ce qui signifie que le modèle prédit parfaitement les valeurs d'entraînement. Cependant, les performances sur le jeu de test sont nettement moins bonnes, avec un score de 0.604 et une MSE de 1387.84. L'erreur absolue moyenne (MAE) sur le jeu de test est de 18.30.

    Ces résultats suggèrent que le modèle souffre de surapprentissage (overfitting). En d'autres termes, il s'ajuste trop précisément aux données d'entraînement et ne généralise pas bien sur les nouvelles données. 
    """)

    st.write("#### 2/ Le XGBoost")

    st.write("###### Résultats :")
    st.write("""
    - **Score du jeu d'entraînement du modèle utilisant le XGBRegressor** : 0.816
    - **Score du jeu de test du modèle utilisant le XGBRegressor** : 0.635
    - **Erreur quadratique moyenne (MSE) sur les données d'entraînement avec XGB** : 815.71
    - **Erreur quadratique moyenne (MSE) sur les données de test avec XGB** : 1279.51
    - **Erreur absolue moyenne (MAE) sur les données d'entraînement avec XGB** : 15.31
    - **Erreur absolue moyenne (MAE) sur les données de test avec XGB** : 20.55
    """)

    st.write("""
    ###### Interprétation :
    Les résultats obtenus avec le modèle XGBRegressor montrent des performances raisonnablement bonnes sur les données d'entraînement avec un score de 0.816 et une erreur quadratique moyenne (MSE) de 815.71. Cela indique que le modèle est capable de bien s'ajuster aux données d'entraînement.

    Cependant, les performances sur le jeu de test sont légèrement inférieures avec un score de 0.635 et une MSE de 1279.51. L'erreur absolue moyenne (MAE) sur le jeu de test est de 20.55.

    Ces résultats suggèrent que le modèle XGBRegressor, bien qu'il soit performant, pourrait encore être amélioré pour mieux généraliser aux nouvelles données. 
    """)

    st.write("#### 3/ Le RandomForest Regressor")

    st.write("###### Résultats :")
    st.write("""
    - **Score du jeu d'entraînement du modèle utilisant le RandomForest** : 0.982
    - **Score du jeu de test du modèle utilisant le RandomForest** : 0.711
    - **Erreur quadratique moyenne (MSE) sur les données d'entraînement avec RandomForest** : 81.11
    - **Erreur quadratique moyenne (MSE) sur les données de test avec RandomForest** : 1012.11
    - **Erreur absolue moyenne (MAE) sur les données d'entraînement avec RandomForest** : 4.12
    - **Erreur absolue moyenne (MAE) sur les données de test avec RandomForest** : 16.09
    """)

    st.write("""
    ###### Interprétation :
    Les résultats obtenus avec le modèle RandomForest Regressor montrent des performances très élevées sur les données d'entraînement avec un score de 0.982 et une erreur quadratique moyenne (MSE) de 81.11. Cela indique que le modèle s'ajuste extrêmement bien aux données d'entraînement.

    Cependant, les performances sur le jeu de test, bien qu'acceptables, sont moins impressionnantes avec un score de 0.711 et une MSE de 1012.11. L'erreur absolue moyenne (MAE) sur le jeu de test est de 16.09.

    Ces résultats suggèrent que le modèle RandomForest Regressor pourrait bénéficier d'améliorations pour mieux généraliser aux nouvelles données. 
    """)

    st.write("#### 4/ Le Gradient Boosting Regressor")

    st.write("###### Résultats :")
    st.write("""
    - **Score du jeu d'entraînement du modèle utilisant le GradientBoostingRegressor** : 0.413
    - **Score du jeu de test du modèle utilisant le GradientBoostingRegressor** : 0.413
    - **Erreur quadratique moyenne (MSE) sur les données d'entraînement avec GradientBoostingRegressor** : 2602.62
    - **Erreur quadratique moyenne (MSE) sur les données de test avec GradientBoostingRegressor** : 2060.56
    - **Erreur absolue moyenne (MAE) sur les données d'entraînement avec GradientBoostingRegressor** : 29.484025861707728
    - **Erreur absolue moyenne (MAE) sur les données de test avec GradientBoostingRegressor** : 28.872413273515676
    """)

    st.write("""
    ###### Interprétation :
    Les résultats obtenus avec le modèle GradientBoostingRegressor montrent des performances modérées sur les données d'entraînement avec un score de 0.413 et une erreur quadratique moyenne (MSE) de 2602.62. Cela indique que le modèle s'ajuste mal aux données d'entraînement.

    Les performances sur le jeu de test sont similaires avec un score de 0.413 et une MSE de 2060.56. L'erreur absolue moyenne (MAE) sur le jeu de test est de 28.87.

    Ces résultats suggèrent que le modèle GradientBoostingRegressor ne correspond pas à notre problématique et ne sera donc pas séléctionné comme modèle retenu.
    """)

    st.write("## Le modèle sélectionné : le Random Forest Regressor")

    st.write("""Au vu des résultats des tests des différents modèle, j'ai choisi le Random Forest Regressor qui me semble être le plus adapté à la problématique.
            Pour optimiser son fonctionnement, j'ai utilisé un GridSearchCV. Voici, son code et ses résultats :""")
    
    st.write("##### Le code :")
    
    st.code("""from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel

    # Définition des valeurs à tester pour chaque hyperparamètre
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    # Initialisation du modèle
    rf = RandomForestRegressor(random_state=42)

    # Initialisation de GridSearchCV avec le modèle et la grille d'hyperparamètres
    regression_model_random_forest = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    # Entraînement du modèle avec GridSearchCV
    regression_model_random_forest.fit(X_train, y_train)

    # Évaluation du modèle sur les données de test
    best_rf = regression_model_random_forest.best_estimator_
    test_score = best_rf.score(X_test, y_test)
    print(f"Meilleur score sur les données de test : {test_score}")

    # Utilisation de la validation croisée pour une évaluation plus robuste
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)
    print(f"Scores de validation croisée : {cv_scores}")
    print(f"Moyenne des scores de validation croisée : {cv_scores.mean()}")
    """)

    st.write("#####  Les résultats :")

    st.write("""
    - **Score du jeu d'entraînement du modèle utilisant le GradientBoostingRegressor** : 0.9999990830343426
    - **Score du jeu de test du modèle utilisant le GradientBoostingRegressor** : 0.7747517995950621
    - **Erreur quadratique moyenne (MSE) sur les données d'entraînement avec GradientBoostingRegressor** : 0.004063076386127533
    - **Erreur quadratique moyenne (MSE) sur les données de test avec GradientBoostingRegressor** : 790.0618651281067
    - **Erreur absolue moyenne (MAE) sur les données d'entraînement avec GradientBoostingRegressor** : 0.0016180217026652648
    - **Erreur absolue moyenne (MAE) sur les données de test avec GradientBoostingRegressor** : 14.645880811406819
    """)

    st.write("Voyons maintenant une illustration graphique des prédictions par rapport à ce qu'il s'est passé réelement :")

    y_test = lire_csv('y_test.csv', sep=',')
    pred_test = lire_csv('pred_test.csv', sep=',')

    # Nombre total de valeurs à afficher
    num_values_to_plot = 7 * 24

    # Convertir y_test et pred_test en arrays numpy
    y_test_array = np.array(y_test)
    pred_test_array = np.array(pred_test)

    # Sélectionner uniquement les 7*24 premières valeurs
    y_test_array_reduced = y_test_array[:num_values_to_plot]
    pred_test_array_reduced = pred_test_array[:num_values_to_plot]

    # Créer un index pour les données réduites
    index_reduced = np.arange(num_values_to_plot)

    # Créer la figure et les axes
    fig, ax = plt.subplots(figsize=(14, 7))

    # Tracer les valeurs réelles
    ax.plot(index_reduced, y_test_array_reduced, label='Valeurs Réelles', color='blue')

    # Tracer les valeurs prédites
    ax.plot(index_reduced, pred_test_array_reduced, label='Valeurs Prédites', color='red', linestyle='--')

    # Ajouter des étiquettes, une légende et un titre
    ax.set_xlabel('Index')
    ax.set_ylabel('Valeur')
    ax.set_title('Comparaison entre Valeurs Prédites et Valeurs Réelles (7 jours)')
    ax.legend()

    # Afficher le graphique avec Streamlit
    st.pyplot(fig)

if page == pages[4]:
    st.write("## 🚴‍♂️ Conclusion 🚴‍♀️")

    st.write("")
    st.write("")

    st.write("""
             

Nous pouvons conclure de nos expérimentations que le Random Forest Regressor semble apporter de bons résultats pour notre problème de machine learning. Pourtant, malgré ces résultats encourageants, la visualisation des prédictions faites avec ce modèle nous montre ses limites : la prédiction semble fiable pendant les heures creuses mais moins précise pendant les heures de pointe, lorsque le nombre de vélos comptés augmente significativement.

Comme énoncé précédemment, pour améliorer le modèle, j'aurais pu ajouter des variables explicatives à mon dataset, telles que les vacances scolaires ou les jours de grève.

L'utilisation de GridSearchCV m'a permis d'améliorer sensiblement mon modèle. Cependant, j'ai utilisé un nombre réduit de données (environ 150,000 lignes) et le GridSearch a mis plusieurs heures à s'exécuter. Je me demande alors s'il serait cohérent de l'utiliser sur un volume de données beaucoup plus important.

Pour de futures améliorations, il serait pertinent de considérer des techniques d'optimisation plus avancées, comme RandomizedSearchCV, ou des approches plus rapides comme l'utilisation de modèles de machine learning distribués.
    """)

