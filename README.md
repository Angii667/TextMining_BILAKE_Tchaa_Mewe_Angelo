# Text Mining — Lexique Sara & Analyse de Textes Médicaux

## Structure du projet

```
TextMining_BLK/
├── README.md
├── requirements.txt
├── SUJET_EXAMEN.md
├── exam/
│   ├── extraction.py     # Partie A — Extraction du lexique Sara depuis le PDF
│   ├── exploration.py    # Partie B — Exploration statistique du lexique Sara
│   └── analyse.py        # Partie C — Analyse avancée (dataset Medical Text)
├── data/
│   ├── raw/              # PDF source (non publié)
│   ├── sara_wide.csv     # Format large : 1 ligne/terme français, 1 colonne/dialecte
│   ├── sara_long.csv     # Format long : 1 ligne/paire de traduction
│   ├── pairs/            # 14 fichiers CSV (1 par dialecte)
│   └── medical_text.csv  # Dataset Medical Text (étiquettes 1-5)
└── figures/
    ├── paires_par_dialecte.png
    ├── heatmap_couverture.png
    ├── longueur_mots_boxplot.png
    ├── wordcloud_Mb.png
    ├── wordcloud_Ngb.png
    └── partie_c/
        ├── lda_perplexite.png
        ├── lda_distribution_themes.png
        ├── confusion_*.png
        ├── comparaison_classifieurs.png
        ├── kmeans_coude.png
        ├── kmeans_silhouette.png
        ├── dendrogramme.png
        └── clusters_2d_pca.png
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

## Exécution

### Partie A — Extraction
```bash
python exam/extraction.py
```
Produit : `sara_wide.csv`, `sara_long.csv`, et 14 fichiers dans `data/pairs/`

### Partie B — Exploration
```bash
python exam/exploration.py
```
Affiche les statistiques et génère les graphiques dans `figures/`

### Partie C — Analyse avancée
```bash
python exam/analyse.py
```
Applique Topic Modeling (LDA), Classification supervisée (NB, LR, SVM) et Clustering (K-Means, Hiérarchique) sur le dataset Medical Text.

## Dataset Partie C

Dataset : **Medical Text** (Kaggle) — 14 438 résumés médicaux classés en 5 catégories :
1. Maladies de l'appareil digestif
2. Maladies cardiovasculaires
3. Tumeurs
4. Maladies du système nerveux
5. Affections pathologiques générales

## Auteur

Projet réalisé dans le cadre du cours de Text Mining.
