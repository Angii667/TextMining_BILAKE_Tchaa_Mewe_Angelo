# Text Mining Multilingue et Analyse de Textes Médicaux

Projet de **fouille de textes** divisé en trois parties complémentaires : extraction d'un lexique multilingue Sara depuis un PDF hérité, exploration statistique d'un corpus de 14 dialectes, et pipeline NLP complet sur un corpus médical PubMed.

## Structure du projet

```
TextMining_BILAKE_Tchaa_Mewe_Angelo/
├── rapport_TextMining.pdf       # Rapport final (15 pages)
├── rapport_TextMining.tex       # Sources LaTeX du rapport
├── requirements.txt             # Dépendances Python
├── exam/
│   ├── extraction.py            # Partie A — OCR Tesseract (pipeline principal)
│   ├── extraction_gemini.py     # Partie A — Variante API Gemini 2.5 Flash
│   ├── exploration.py           # Partie B — Exploration statistique
│   └── analyse.py               # Partie C — NLP médical (LDA, classif., clustering)
├── data/
│   ├── raw/SaraLanguagesLexicon.pdf   # Dictionnaire Sara-Français source
│   ├── sara_wide.csv                 # 1104 termes français × 14 dialectes
│   ├── sara_long.csv                 # 10 600 paires de traduction
│   ├── pairs/                        # 14 fichiers bilingues individuels
│   ├── train.dat                     # Corpus médical (entraînement)
│   └── test.dat                      # Corpus médical (test)
├── images/
│   ├── armoirie.jpg                  # Armoiries du Togo
│   └── logo_ecole.jpeg               # Logo EPL
└── figures/
    ├── paires_par_dialecte.png
    ├── heatmap_couverture.png
    ├── longueur_mots_boxplot.png
    ├── wordcloud_Mb.png
    ├── wordcloud_Ngb.png
    └── partie_c/
        ├── lda_coherence.png
        ├── clusters_2d_umap.png
        ├── comparaison_classifieurs.png
        ├── dendrogramme.png
        └── confusion_*.png
```

## Problème

Le **SaraLanguagesLexicon.pdf** utilise des polices héritées dont les tables Unicode (ToUnicode CMap) sont absentes ou corrompues. Toute extraction native (pdfplumber, PyMuPDF, pdfminer) produit du texte corrompu pour les caractères de l'Alphabet National Tchadien (ɓ, ɗ, ɛ, ɔ, ɨ, ə, etc.).

**Solution adoptée :** pipeline OCR complet — conversion PDF → image (300 DPI) → Tesseract LSTM → parsing post-OCR. Une tentative complémentaire avec l'API Gemini 2.5 Flash Vision a été explorée (`extraction_gemini.py`).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

**Dépendances système :** `tesseract-ocr`, `tesseract-ocr-fra`, `poppler-utils`

## Exécution

### Partie A — Extraction du lexique Sara
```bash
python exam/extraction.py
```
Produit : `sara_wide.csv`, `sara_long.csv`, 14 fichiers dans `data/pairs/`

Variante Gemini (nécessite `GEMINI_API_KEY`) :
```bash
export GEMINI_API_KEY='votre_clé'
python exam/extraction_gemini.py
```

### Partie B — Exploration statistique
```bash
python exam/exploration.py
```
Génère les graphiques dans `figures/`

### Partie C — Pipeline NLP médical
```bash
python exam/analyse.py
```
Pipeline complet : prétraitement, LDA (k=5..20), classification supervisée (GridSearchCV sur 4 modèles), clustering (K-Means + hiérarchique avec visualisation UMAP).

## Partie A — Extraction du lexique

| Approche | Résultat |
|----------|----------|
| pdfplumber / PyMuPDF / pdfminer | Texte corrompu (polices legacy sans ToUnicode) |
| Mapping auto via XML_Data.zip | Échec : corruption non systématique |
| OCR Tesseract LSTM (300 DPI) | **Adopté** — ~1104 termes, 10 600 paires, table de correction de ~10 entrées |
| Gemini 2.5 Flash Vision | Qualité supérieure mais limité par quotas API (503) |

## Partie B — Exploration

- Couverture asymétrique : Ngambay et Mbay ≈ 100 %, Bediondo et Nar < 50 %
- Longueur moyenne des mots : 4-6 caractères (Sara) vs 7-9 (français)
- Heatmap binaire et nuages de mots pour les dialectes majoritaires

## Partie C — NLP Médical

**Dataset :** 14 438 résumés PubMed — 5 classes déséquilibrées

| Modèle | Accuracy | F1-macro |
|--------|----------|----------|
| SVM linéaire | 0,601 | 0,544 |
| Naive Bayes | 0,591 | 0,534 |
| Régression logistique | 0,697 | 0,649 |
| Forêt aléatoire | 0,459 | 0,340 |

**Topic Modeling :** LDA optimal à k=10 (cohérence cv = 0,48)
**Clustering :** Visualisation UMAP + dendrogramme hiérarchique

## Auteur

**BILAKE Tchaa Mèwè Angelo**  
Licence Fondamentale Intelligence Artificielle & Big Data  
Université de Lomé — École Polytechnique de Lomé  
MTH1621: Data Mining — Année 2025-2026

Sous la supervision de **Donald TITEMBAYE**.
