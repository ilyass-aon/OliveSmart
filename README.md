# OliveSmart — Automated ML/DL Framework for Olive Yield Prediction
# Project: OLV-5 | University Project

## 🫒 Présentation

**OliveSmart** est un framework expérimental Python modulaire conçu pour prédire le **rendement des oliveraies (kg/ha)** à partir de données hétérogènes :
- Données historiques FAOSTAT
- Données climatiques NASA POWER
- Indices de végétation Sentinel-2 (NDVI)
- Données hydriques FAO WaPOR

---

## 🗂️ Structure du Projet

```
OliveSmart/
├── main.py                    # Point d'entrée — une seule commande
├── config.py                  # Configuration centrale
├── requirements.txt           # Dépendances ML
├── requirements_deep.txt      # Dépendances Deep Learning
│
├── data/
│   ├── raw/                   # ← PLACER VOS DONNÉES ICI
│   │   ├── faostat/           # CSV exporté depuis fao.org/faostat
│   │   ├── nasa_power/        # CSV depuis power.larc.nasa.gov
│   │   ├── sentinel2/         # CSV NDVI extrait manuellement
│   │   └── wapor/             # CSV depuis FAO WaPOR
│   └── processed/             # Données transformées (auto-généré)
│
├── src/
│   ├── data/                  # Chargement, EDA, prétraitement, découpage
│   ├── models/                # Baseline, Ridge, RF, XGBoost, MLP, TFT
│   ├── optimization/          # GridSearch, RandomSearch, Optuna
│   ├── experiment_runner.py   # Orchestrateur séquentiel
│   ├── evaluator.py           # Métriques MAE, RMSE, R²
│   ├── visualizer.py          # Graphiques automatiques
│   ├── model_selector.py      # Sélection du meilleur modèle
│   └── report_generator.py   # Rapports PDF et CSV
│
├── results/                   # Résultats des expériences (auto-généré)
├── reports/                   # Rapports PDF et CSV (auto-généré)
└── logs/                      # Fichiers de logs horodatés (auto-généré)
```

---

## 🚀 Installation

### 1. Cloner le projet
```bash
# Si vous utilisez Git
git clone <url_du_repo>
cd olive
```

### 2. Créer un environnement virtuel (recommandé)
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
```

### 3. Installer les dépendances ML de base
```bash
pip install -r requirements.txt
```

### 4. Installer les dépendances Deep Learning (optionnel)
```bash
# Pour GPU NVIDIA avec CUDA 12.x (recommandé pour RTX 5060 Ti)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_deep.txt
```

---

## 📂 Placement des Données

Avant de lancer le pipeline, placez vos fichiers de données dans les bons répertoires :

| Dataset | Répertoire | Format attendu |
|---------|-----------|----------------|
| FAOSTAT | `data/raw/faostat/` | CSV exporté depuis faostat.org |
| NASA POWER | `data/raw/nasa_power/` | CSV depuis power.larc.nasa.gov |
| Sentinel-2 | `data/raw/sentinel2/` | CSV avec colonnes date + NDVI |
| FAO WaPOR | `data/raw/wapor/` | CSV avec colonnes date + ET |

---

## ▶️ Utilisation

```bash
# Lancer tout le pipeline (tous les datasets, tous les modèles)
python main.py

# Vérifier l'environnement uniquement (Phase 1)
python main.py --phase 1

# Traiter un seul dataset
python main.py --dataset FAOSTAT

# Tester un seul modèle sur tous les datasets
python main.py --model Ridge

# Combinaison spécifique
python main.py --dataset FAOSTAT --model XGBoost

# Relancer tout depuis zéro (ignorer le cache)
python main.py --force-rerun

# Simuler sans exécuter
python main.py --dry-run

# Lister les datasets et modèles disponibles
python main.py --list
```

---

## ⚙️ Configuration

Toute la configuration se trouve dans [`config.py`](config.py).

Paramètres principaux modifiables :
- `DATASETS` : activer/désactiver chaque source de données
- `MODELS` : activer/désactiver chaque modèle
- `OPTIMIZATION.optuna_n_trials` : nombre d'essais Optuna
- `OPTIMIZATION.max_time_per_model_s` : durée max par modèle
- `SPLITTING` : ratios train/val/test
- `REPORTING.ranking_metric` : critère de sélection du meilleur modèle

---

## 📊 Modèles Implémentés

| Modèle | Bibliothèque | Hyperparamètres optimisés |
|--------|-------------|--------------------------|
| Baseline | — | Aucun |
| Ridge | scikit-learn | alpha, fit_intercept, solver |
| Random Forest | scikit-learn | n_estimators, max_depth, min_samples_* |
| XGBoost | xgboost | learning_rate, n_estimators, max_depth, subsample |
| MLP | TensorFlow/Keras | architecture, learning_rate, dropout, batch_size |
| TFT/LSTM | PyTorch | learning_rate, hidden_size, dropout (si données suffisantes) |

---

## 📋 Contraintes Scientifiques

> ⚠️ Ces règles sont strictement appliquées par le framework :

1. **Pas de fuite de données** : les transformations sont ajustées uniquement sur les données d'entraînement
2. **Découpage chronologique** : pour les séries temporelles, l'ordre temporel est toujours respecté
3. **Optimisation sur validation** : le jeu de test n'est jamais utilisé pendant la recherche d'hyperparamètres
4. **Indépendance des datasets** : chaque dataset est traité séparément
5. **Reproductibilité** : graine aléatoire fixe dans toutes les opérations stochastiques
6. **Pas d'invention de résultats** : si un dataset n'a pas de variable cible valide, seule l'analyse exploratoire est réalisée

---

## 📁 Sorties Générées

```
reports/
├── FAOSTAT/
│   ├── rapport_FAOSTAT.pdf           # Rapport détaillé
│   ├── experiments_FAOSTAT.csv       # Toutes les expériences
│   └── graphs/                        # Graphiques
├── NASA_POWER/
│   └── rapport_NASA_POWER.pdf        # Rapport exploratoire
├── BEST_MODELS_COMPARISON.csv        # Tableau comparatif global
└── OLIVESMART_GLOBAL_REPORT.pdf      # Rapport complet du projet
```

---

## 🔄 Reprise d'Exécution

Si le pipeline est interrompu, relancez simplement :
```bash
python main.py
```

Les expériences déjà terminées sont automatiquement ignorées (fichiers JSON dans `results/experiments/`).

Pour forcer la réexécution de tout :
```bash
python main.py --force-rerun
```

---

## 📚 Références

- FAOSTAT : https://www.fao.org/faostat/en/#data/QCL
- NASA POWER : https://power.larc.nasa.gov/
- Sentinel-2 (GEE) : https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
- FAO WaPOR : https://www.fao.org/in-action/remote-sensing-for-water-productivity/

---

*Projet universitaire OLV-5 — OliveSmart | Framework développé progressivement en 8 phases*
