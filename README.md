# Rapport comparatif : Gestion des données et méthodologie IA

## Introduction
Ce rapport compare la gestion des données et la méthodologie d'intelligence artificielle de quatre projets :
- **[movie-recommender](https://github.com/Nano-a/movie-recommender)**
- **[medicinal-plant-classifier](https://github.com/Nano-a/medicinal-plant-classifier)**
- **[traffic-flow-predictor](https://github.com/Nano-a/traffic-flow-predictor)**
- **[energy-demand-prediction](https://github.com/Nano-a/energy-demand-prediction)**

L'objectif est d'analyser, dossier par dossier et fichier par fichier, la façon dont chaque projet traite les données et implémente l'IA, en mettant en avant les différences et similitudes, sans s'attarder sur le sujet traité.

---

## 1. [movie-recommender](https://github.com/Nano-a/movie-recommender)

### a. Chargement et gestion des données
- **Fichiers utilisés** : `data/movies.csv`, `data/ratings.csv` (voir [model/recommender.py](movie-recommender/model/recommender.py) lignes 7-11)
- **Chargement** : Utilisation de `pandas.read_csv` pour charger les jeux de données réels (films et notes utilisateurs).
- **Prétraitement** :
  - Filtrage des utilisateurs ayant noté au moins 5 films (lignes 15-22)
  - Création d'une matrice utilisateur-film (pivot, valeurs manquantes à 0)
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
- [README.md](movie-recommender/README.md) sections "Fonctionnement & points d'entrée" et "Données"

---

## 2. [medicinal-plant-classifier](https://github.com/Nano-a/medicinal-plant-classifier)

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
- [README.md](medicinal-plant-classifier/README.md) sections "Fonctionnement & points d'entrée" et "Données"

---

## 3. [traffic-flow-predictor](https://github.com/Nano-a/traffic-flow-predictor)

### a. Chargement et gestion des données
- **Fichiers utilisés** : Données synthétiques générées à la volée, pas de fichier CSV par défaut ([model/traffic_model.py](traffic-flow-predictor/model/traffic_model.py) lignes 13-56)
- **Chargement** : Génération de données simulées (heure, jour, météo, volume) ([model/traffic_model.py](traffic-flow-predictor/model/traffic_model.py) lignes 21-56)
- **Prétraitement** :
  - Encodage des variables catégorielles avec `LabelEncoder` (lignes 57-65)
  - Pas de normalisation car toutes les features sont numériques ou discrètes
- **Séparation train/test** : Non explicitement séparé, car tout est généré et utilisé pour l'entraînement et l'évaluation interne

### b. Méthodologie IA
- **Modèle** : RandomForestClassifier (lignes 14-20)
- **Évaluation** : Accuracy, importance des variables ([model/traffic_model.py](traffic-flow-predictor/model/traffic_model.py) lignes 87-110)
- **Visualisation** : Statistiques affichées, possibilité d'ajouter des graphiques (voir README)

### c. Références
- [model/traffic_model.py](traffic-flow-predictor/model/traffic_model.py)
- [README.md](traffic-flow-predictor/README.md) sections "Fonctionnement & points d'entrée" et "Données"

---

## 4. [energy-demand-prediction](https://github.com/Nano-a/energy-demand-prediction)

### a. Chargement et gestion des données
- **Fichiers utilisés** : `city-hall-electricity-use.csv` (données réelles de Boston) ([preprocessing.py](energy-demand-prediction/preprocessing.py) lignes 6-7)
- **Chargement** : Utilisation de `pandas.read_csv` avec parsing des dates pour charger les données de consommation électrique.
- **Prétraitement** :
  - Nettoyage des valeurs nulles et aberrantes (lignes 10-12)
  - Suppression des doublons avec agrégation par timestamp (lignes 14-15)
  - Interpolation linéaire pour les valeurs manquantes (lignes 20-25)
  - Feature engineering temporel (heure, jour, mois, année, week-end) (lignes 28-35)
  - Création de lags temporels (15min, 1h, 1 jour) (lignes 37-40)
  - Standardisation des features avec `StandardScaler` (lignes 45-47)
- **Séparation train/test** : Split 70/15/15 (train/validation/test) ([preprocessing.py](energy-demand-prediction/preprocessing.py) lignes 49-56)

### b. Méthodologie IA
- **Baseline** : Régression linéaire et ARIMA ([model_baseline.py](energy-demand-prediction/model_baseline.py) lignes 18-25 et 30-45)
- **Machine Learning** : RandomForest, XGBoost, LightGBM ([model_ml.py](energy-demand-prediction/model_ml.py) lignes 22-60)
- **Deep Learning** : LSTM avec séquences temporelles ([model_lstm.py](energy-demand-prediction/model_lstm.py) lignes 25-40)
- **Optimisation** : GridSearchCV pour tuning d'hyperparamètres ([optimisation.py](energy-demand-prediction/optimisation.py) lignes 15-25)
- **Évaluation** : RMSE, MAE, MAPE sur tous les modèles
- **Interface** : Application Streamlit complète avec visualisations interactives ([app.py](energy-demand-prediction/app.py))

### c. Références
- [preprocessing.py](energy-demand-prediction/preprocessing.py)
- [model_baseline.py](energy-demand-prediction/model_baseline.py)
- [model_ml.py](energy-demand-prediction/model_ml.py)
- [model_lstm.py](energy-demand-prediction/model_lstm.py)
- [optimisation.py](energy-demand-prediction/optimisation.py)
- [app.py](energy-demand-prediction/app.py)
- [README.md](energy-demand-prediction/README.md) sections "Pipeline de traitement" et "Résultats"

---

## 5. Tableau comparatif synthétique

| Projet                      | Source des données         | Prétraitement         | Séparation train/test | Modèle IA                | Évaluation                |
|-----------------------------|---------------------------|-----------------------|----------------------|--------------------------|---------------------------|
| movie-recommender           | Fichiers CSV réels        | Pivot, one-hot, filtres| Oui                  | SVD, cosinus             | RMSE, visualisations      |
| medicinal-plant-classifier  | Fichier CSV réel          | Encodage, normalisation| Oui                  | RandomForest             | Accuracy, F1, ROC, etc.   |
| traffic-flow-predictor      | Données synthétiques      | Encodage              | Non (auto-éval)      | RandomForest             | Accuracy, importance      |
| energy-demand-prediction    | Fichier CSV réel          | Feature engineering, standardisation| Oui (70/15/15) | Multiple (LR, RF, XGB, LGBM, LSTM) | RMSE, MAE, MAPE |

---

## 6. Conclusion
- **[movie-recommender](https://github.com/Nano-a/movie-recommender)** utilise des données réelles, un double mode de recommandation (collaboratif et contenu), et une évaluation quantitative (RMSE).
- **[medicinal-plant-classifier](https://github.com/Nano-a/medicinal-plant-classifier)** repose sur un jeu de données réel, un prétraitement complet (encodage + normalisation), et une évaluation multi-métriques.
- **[traffic-flow-predictor](https://github.com/Nano-a/traffic-flow-predictor)** se distingue par la génération de données synthétiques, un prétraitement minimal, et une évaluation simple mais automatisée.
- **[energy-demand-prediction](https://github.com/Nano-a/energy-demand-prediction)** se caractérise par un pipeline complet de data science avec feature engineering temporel, multiples modèles (baseline, ML, deep learning), optimisation d'hyperparamètres, et une interface web Streamlit complète.