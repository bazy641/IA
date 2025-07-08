# Rapport comparatif : Gestion des données et méthodologie IA

## Introduction
Ce rapport compare la gestion des données et la méthodologie d’intelligence artificielle de trois projets :
- **movie-recommender**
- **medicinal-plant-classifier**
- **traffic-flow-predictor**

L’objectif est d’analyser, dossier par dossier et fichier par fichier, la façon dont chaque projet traite les données et implémente l’IA, en mettant en avant les différences et similitudes, sans s’attarder sur le sujet traité.

---

## 1. movie-recommender

### a. Chargement et gestion des données
- **Fichiers utilisés** : `data/movies.csv`, `data/ratings.csv` (voir [model/recommender.py](movie-recommender/model/recommender.py) lignes 7-11)
- **Chargement** : Utilisation de `pandas.read_csv` pour charger les jeux de données réels (films et notes utilisateurs).
- **Prétraitement** :
  - Filtrage des utilisateurs ayant noté au moins 5 films (lignes 15-22)
  - Création d’une matrice utilisateur-film (pivot, valeurs manquantes à 0)
  - Filtrage des films sans notes (lignes 27-30)
  - Encodage des genres de films en one-hot (lignes 32-34)
- **Séparation train/test** : Fonction dédiée dans [model/evaluation.py](movie-recommender/model/evaluation.py) lignes 6-10

### b. Méthodologie IA
- **Collaboratif** : SVD (Singular Value Decomposition) via `TruncatedSVD` de scikit-learn (lignes 37-44)
- **Contenu** : Similarité cosinus sur les genres (lignes 61-81)
- **Évaluation** : RMSE calculé sur les prédictions ([model/evaluation.py](movie-recommender/model/evaluation.py) lignes 12-38)
- **Visualisation** : Génération de graphiques ([visualizations/plots.py](movie-recommender/visualizations/plots.py))

### c. Références
- [model/recommender.py](movie-recommender/model/recommender.py)
- [model/evaluation.py](movie-recommender/model/evaluation.py)
- [README.md](movie-recommender/README.md) sections "Fonctionnement & points d’entrée" et "Données"

---

## 2. medicinal-plant-classifier

### a. Chargement et gestion des données
- **Fichiers utilisés** : `data/plants.csv` (et éventuellement `plants_extended.csv`) ([model/plant_classifier.py](medicinal-plant-classifier/model/plant_classifier.py) lignes 29-31)
- **Chargement** : Utilisation de `pandas.read_csv` pour charger les données botaniques.
- **Prétraitement** :
  - Encodage des variables catégorielles avec `LabelEncoder` ([model/plant_classifier.py](medicinal-plant-classifier/model/plant_classifier.py) lignes 47-67)
  - Normalisation des variables numériques avec `StandardScaler` (lignes 68-71)
  - Séparation des features et de la cible (lignes 164-181)
- **Séparation train/test** : Utilisation de `train_test_split` de scikit-learn (lignes 324-332)

### b. Méthodologie IA
- **Modèle** : RandomForestClassifier (lignes 72-97)
- **Explicabilité** : Importance des features calculée et affichée (lignes 192-220)
- **Évaluation** : Accuracy, precision, recall, F1-score, confusion matrix, ROC AUC ([model/plant_classifier.py](medicinal-plant-classifier/model/plant_classifier.py) lignes 273-323)
- **Visualisation** : Génération automatique de graphiques (matrice de confusion, distribution, ROC, importance) (lignes 192-240)

### c. Références
- [model/plant_classifier.py](medicinal-plant-classifier/model/plant_classifier.py)
- [README.md](medicinal-plant-classifier/README.md) sections "Fonctionnement & points d’entrée" et "Données"

---

## 3. traffic-flow-predictor

### a. Chargement et gestion des données
- **Fichiers utilisés** : Données synthétiques générées à la volée, pas de fichier CSV par défaut ([model/traffic_model.py](traffic-flow-predictor/model/traffic_model.py) lignes 13-56)
- **Chargement** : Génération de données simulées (heure, jour, météo, volume) ([model/traffic_model.py](traffic-flow-predictor/model/traffic_model.py) lignes 21-56)
- **Prétraitement** :
  - Encodage des variables catégorielles avec `LabelEncoder` (lignes 57-65)
  - Pas de normalisation car toutes les features sont numériques ou discrètes
- **Séparation train/test** : Non explicitement séparé, car tout est généré et utilisé pour l’entraînement et l’évaluation interne

### b. Méthodologie IA
- **Modèle** : RandomForestClassifier (lignes 14-20)
- **Évaluation** : Accuracy, importance des variables ([model/traffic_model.py](traffic-flow-predictor/model/traffic_model.py) lignes 87-110)
- **Visualisation** : Statistiques affichées, possibilité d’ajouter des graphiques (voir README)

### c. Références
- [model/traffic_model.py](traffic-flow-predictor/model/traffic_model.py)
- [README.md](traffic-flow-predictor/README.md) sections "Fonctionnement & points d’entrée" et "Données"

---

## 4. Tableau comparatif synthétique

| Projet                      | Source des données         | Prétraitement         | Séparation train/test | Modèle IA                | Évaluation                |
|-----------------------------|---------------------------|-----------------------|----------------------|--------------------------|---------------------------|
| movie-recommender           | Fichiers CSV réels        | Pivot, one-hot, filtres| Oui                  | SVD, cosinus             | RMSE, visualisations      |
| medicinal-plant-classifier  | Fichier CSV réel          | Encodage, normalisation| Oui                  | RandomForest             | Accuracy, F1, ROC, etc.   |
| traffic-flow-predictor      | Données synthétiques      | Encodage              | Non (auto-éval)      | RandomForest             | Accuracy, importance      |

---

## 5. Conclusion
- **movie-recommender** utilise des données réelles, un double mode de recommandation (collaboratif et contenu), et une évaluation quantitative (RMSE).
- **medicinal-plant-classifier** repose sur un jeu de données réel, un prétraitement complet (encodage + normalisation), et une évaluation multi-métriques.
- **traffic-flow-predictor** se distingue par la génération de données synthétiques, un prétraitement minimal, et une évaluation simple mais automatisée.

Pour chaque détail, tu peux te référer aux fichiers et lignes indiqués pour vérifier la gestion des données et la méthodologie IA.

---

*Rapport généré automatiquement. Pour toute question, consulte les fichiers référencés ou demande une analyse plus approfondie sur une partie précise.* 