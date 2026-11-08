# OliveSmart

Framework Python expérimental pour préparer des données agricoles, explorer
des indicateurs et, à terme, prédire le rendement des oliveraies au Maroc.
Le projet OLV-5 combine des données FAOSTAT, NASA POWER, Sentinel-2/NDVI et
FAO WaPOR.

> **État du projet — 8 novembre 2026 :** les phases 1 à 4 sont présentes.
> L'entraînement ML classique et la comparaison des résultats sont disponibles.
> Les phases deep learning et les rapports finaux restent à implémenter.

## Avancement

| Phase | État | Ce qui est disponible |
|---|---|---|
| 1. Environnement | Implémentée | Vérification des dépendances principales, des données et détection facultative du GPU. |
| 2. Chargement et EDA | Implémentée | Chargeurs pour les quatre sources, métadonnées des fichiers, analyse exploratoire, graphiques et rapports JSON/CSV. |
| 3. Prétraitement et découpage | Implémentée, principalement pour FAOSTAT | Préparation de la cible FAOSTAT, exclusion de la production comme prédicteur, ajout d'un retard de rendement, pipelines de prétraitement et partitions chronologiques. |
| 4. Modèles ML | Implémentée | Baseline, Ridge, Random Forest et XGBoost, validation croisée chronologique, métriques train/validation/test, sauvegarde des modèles et comparaison CSV. |
| 5. MLP | À faire | La configuration existe, mais pas de boucle d'entraînement raccordée au pipeline. |
| 6. TFT/LSTM | À faire | Désactivé par défaut; à évaluer seulement si suffisamment de données temporelles sont disponibles. |
| 7. Évaluation et rapports finaux | Partielle | La phase 4 calcule déjà les métriques et une comparaison CSV; rapports PDF et analyses finales restent à produire. |
| 8. Pipeline complet | Partiel | Les phases 1 à 4 s'exécutent; les phases 5 à 7 restent à développer. |

### Données et résultats présents

- `data/raw/faostat/` contient les fichiers FAOSTAT large et long. Les résultats
  EDA chargés dans le dépôt indiquent 33 observations annuelles de 1990 à 2022
  et une cible `Yield_hg_ha`.
- `data/raw/nasa_power/` contient des données climatiques annuelles et
  mensuelles. Sans rendement associé, elles sont actuellement explorées sans
  apprentissage supervisé.
- `data/raw/sentinel2/` contient des séries NDVI annuelles et mensuelles.
- `data/raw/wapor/` contient une série annuelle d'évapotranspiration.
- Des résultats EDA sont déjà présents pour FAOSTAT et NASA POWER sous
  `results/experiments/eda/`; les métadonnées de partition FAOSTAT sont dans
  `results/experiments/splits/`. Des graphiques existent sous
  `results/graphs/`.
- Les dossiers `reports/` ne contiennent pas encore de rapports finaux générés.

Les données FAOSTAT ne comptent que 33 observations nationales; après création
d'un retard, le découpage enregistré contient 32 observations (22 entraînement,
4 validation, 6 test). Les petites partitions rendent toute métrique très
incertaine. Les données NASA POWER, Sentinel-2 et WaPOR ne sont pas encore
fusionnées avec le rendement dans un jeu de données supervisé commun.

Le téléchargeur prévoit des données de repli dites « représentatives » si
certaines API échouent. Vérifier la provenance et les valeurs effectivement
enregistrées avant toute interprétation scientifique ou publication.

## Fonctionnalités déjà présentes

- Configuration centrale des sources, modèles, partitions et paramètres dans
  `config.py`.
- Chargement des CSV FAOSTAT, NASA POWER, Sentinel-2 et WaPOR, avec détection
  d'encodage, harmonisation de colonnes et métadonnées de chargement.
- Analyse exploratoire : dimensions, types, valeurs manquantes, doublons,
  statistiques descriptives, valeurs aberrantes, corrélations, faisabilité
  supervisée et graphiques.
- Préparation FAOSTAT : tri chronologique, variables de rendement et d'aire,
  retard du rendement précédent; `Production_tonnes` est exclue car elle est
  liée mathématiquement à la cible.
- Prétraitement scikit-learn : imputation, encodage catégoriel et mise à
  l'échelle selon le modèle. Les transformations sont ajustées sur
  l'entraînement puis appliquées à la validation et au test.
- Partition chronologique train/validation/test, sauvegarde de métadonnées et
  indices de partition pour reproductibilité.
- Entraînement des modèles classiques, optimisation par validation croisée
  chronologique et enregistrement des prédictions/métriques de phase 4.
- Téléchargement automatisé disponible via `downloaders/download_all.py`.
- Tests unitaires présents pour l'environnement, les données et le
  prétraitement (`tests/`).

## À faire

1. Vérifier la provenance et la qualité des données téléchargées, en particulier
   lorsque le téléchargeur a utilisé des valeurs de repli.
2. Nettoyer et documenter les données, puis fusionner rendement, climat,
   télédétection et données hydriques sur des périodes et zones compatibles.
3. Implémenter et raccorder l'entraînement des modèles configurés
   deep learning (MLP et, si justifié, TFT/LSTM).
4. Approfondir l'évaluation statistique et temporelle, avec davantage
   d'observations et des intervalles d'incertitude pertinents.
5. Générer les graphiques comparatifs et rapports détaillés; les modèles
   entraînés et la comparaison CSV sont déjà produits par la phase 4.
6. Déterminer si la taille et la granularité des données permettent d'entraîner
   utilement un modèle temporel (TFT/LSTM); le TFT est actuellement désactivé.
7. Exécuter les tests automatisés et compléter les tests à mesure que les
   phases suivantes sont développées.

## Installation

Depuis la racine du dépôt, avec Python 3.10 ou plus récent :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Les dépendances Deep Learning sont séparées et facultatives pour les phases
actuelles :

```powershell
python -m pip install -r requirements_deep.txt
```

Pour CUDA, installer au préalable la version de PyTorch adaptée à la version
CUDA de la machine en suivant les instructions officielles de PyTorch. Le
support GPU n'est pas nécessaire pour lancer les phases implémentées.

## Utilisation

```powershell
# Afficher les datasets et modèles configurés
python main.py --list

# Vérifier l'environnement
python main.py --phase 1

# Charger les données et exécuter l'EDA
python main.py --phase 2

# Prétraiter et produire les partitions supervisées disponibles
python main.py --phase 3

# Entraîner et comparer les modèles ML classiques sur FAOSTAT
python main.py --phase 4 --dataset FAOSTAT

# Télécharger les sources (nécessite une connexion Internet)
python downloaders/download_all.py
```

Les options `--dataset FAOSTAT`, `--model Ridge`, `--force-rerun` et
`--dry-run` existent dans l'interface CLI. La phase 4 entraîne les modèles
classiques sélectionnés (`Baseline`, `Ridge`, `RandomForest`, `XGBoost`).
`--model MLP` et `--model TFT` ne lancent pas encore de modèle. Exécuter
`python main.py` lance la phase 4 après les phases de données, puis indique que
les phases 5 à 7 restent à implémenter.

Les résultats de phase 4 sont enregistrés dans :

- `results/models/<dataset>/` : modèle entraîné (`.joblib`) et paramètres (`.json`);
- `results/experiments/model_results/` : métriques et hyperparamètres par modèle;
- `results/experiments/predictions/` : valeurs prédites pour validation et test;
- `results/experiments/phase4_model_comparison.csv` : classement par métrique de validation.

Les expériences terminées sont réutilisées si les données et les paramètres
correspondent; `--force-rerun` force leur réentraînement.

## Structure actuelle

```text
.
├── config.py
├── main.py
├── downloaders/
│   └── download_all.py
├── data/
│   ├── raw/                 # CSV des quatre sources
│   └── processed/           # Emplacements prévus pour les données traitées
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   ├── data_explorer.py
│   │   ├── preprocessing.py
│   │   └── data_splitter.py
│   ├── models/
│   │   ├── base_model.py
│   │   └── classical_models.py
│   ├── optimization/
│   └── logger.py
├── tests/
├── results/                 # Modèles, EDA, partitions, prédictions et métriques
├── reports/                 # Réservé aux rapports finaux à venir
└── logs/
```

## Sources de données

- [FAOSTAT — Cultures et élevage](https://www.fao.org/faostat/en/#data/QCL)
- [NASA POWER](https://power.larc.nasa.gov/)
- [Sentinel-2 — données Harmonized Sentinel-2 MSI](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- [FAO WaPOR](https://www.fao.org/in-action/remote-sensing-for-water-productivity/)
