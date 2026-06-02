# Examen 

## Contexte

Les langues Sara constituent un groupe de langues parlées principalement au sud du Tchad. Elles appartiennent à la famille Nilo-Saharienne et sont **très peu dotées numériquement** : presque aucune ressource linguistique structurée n'existe pour ces langues sous forme utilisable par les outils informatiques.

Le fichier `SaraLanguagesLexicon.pdf` est un dictionnaire multilingue qui recense environ **2 000 termes français** avec leurs équivalents dans **14 dialectes Sara** : Bebote, Bediondo, Daba, Gor, Gulay, Kaba, Kaba Na, Laka, Mbay, Mango, Nar, Ngambay, Sar, Ngam.

Votre mission se déroule en trois parties :

| Partie | Objet |
|--------|-----------|
| **A — Extraction** | Extraire les données du PDF Sara et produire des CSV |
| **B — Exploration** | Analyser statistiquement le lexique Sara |
| **C — Analyse avancée** |Appliquer 3 méthodes NLP sur votre dataset personnel |

---

## Partie A — Extraction des données Sara (commun)

### A1. Prise en main

Ouvrez `SaraLanguagesLexicon.pdf` et observez :
- Comment les données sont organisées sur la page (colonnes, structure des entrées)
- Comment un terme français et ses traductions sont présentés

Justifiez dans votre rapport le choix de votre bibliothèque d'extraction.

### A2. Implémentation

Complétez la classe `SaraExtractor` dans le fichier `extraction.py`.

Vous devez produire **trois types de fichiers** :

**`sara_wide.csv`** — format large, une ligne par terme français :
```
french_term, Beb, Bd, Db, Gor, Gu, KbN, Kbb, Lk, Mb, Mo, Nar, Ngb, Sr, NgT
chien, ...
eau, ...
```

**`sara_long.csv`** — format long, une ligne par paire de traduction :
```
source_lang, source_word, target_lang, target_word
French, chien, Mbay, ...
French, eau, Gor, ...
```

**Dossier `pairs/`** — un fichier CSV par dialecte, exactement deux colonnes :
```
# mbay_french.csv
mbay, français
...,  chien
...,  eau
```

### A3. Questions (rapport)

1. Quelle bibliothèque avez-vous utilisée pour l'extraction et pourquoi ?
2. Quels problèmes d'encodage avez-vous rencontrés ? Décrivez comment vous les avez détectés et résolus.
3. Quelle stratégie avez-vous utilisée pour séparer les deux colonnes de chaque page ?
4. Comment avez-vous géré les valeurs manquantes (un dialecte sans traduction pour un terme donné) ?
5. Comparez votre approche avec une alternative que vous n'avez pas choisie.

---

## Partie B — Exploration statistique (commun)

Complétez la classe `SaraExplorer` dans le fichier `exploration.py`.

### B1. Statistiques de base

- Nombre total de termes français uniques
- Nombre total de paires de traduction
- Répartition des paires par dialecte
- Taux de couverture par dialecte : quel pourcentage des termes français est traduit ?

### B2. Analyse du vocabulaire

- Distribution de la longueur des mots (en nombre de caractères) par dialecte
- Les 20 termes français présents dans le plus grand nombre de dialectes
- Les 20 termes français présents dans le plus petit nombre de dialectes
- Les termes présents dans **tous** les dialectes (si ils existent)

### B3. Visualisations

- Diagramme en barres du nombre de paires par dialecte
- Boîte à moustaches (boxplot) de la longueur des mots pour au moins 5 dialectes
- Nuage de mots (word cloud) pour au moins 2 dialectes au choix

### B4. Questions
Repondez à ces questions dans votre rapport:

1. Quel dialecte est le mieux représenté ? Le moins représenté ?
2. Certains dialectes semblent-ils plus proches l'un de l'autre d'après leur taux de couverture commun ?
3. Quelle est la longueur moyenne d'un mot Sara ? Comparez avec la longueur moyenne du terme français correspondant.
4. Quelles difficultés pose la représentation des caractères phonétiques (tons, nasalisations, consonnes implosives) pour l'analyse informatique ?

---

## Partie C — Analyse avancée (individuel)

Chaque étudiant travaille sur le **dataset qui lui est assigné** .  
Vous devez appliquer les **trois méthodes** suivantes sur votre dataset : Topic Modeling, Classification supervisée, et Clustering.  
Complétez le fichier `analyse.py`.

### Assignation des datasets

| Étudiant n° | Dataset | Source | Lien |
|-------------|---------|--------|------|
| 1 | BBC News Articles | Kaggle | https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification |
| 2 | AG News | HuggingFace | https://huggingface.co/datasets/ag_news |
| 3 | 20 Newsgroups | scikit-learn | `fetch_20newsgroups(subset='all')` |
| 4 | IMDB Movie Reviews | HuggingFace | https://huggingface.co/datasets/imdb |
| 5 | Yelp Reviews | HuggingFace | https://huggingface.co/datasets/yelp_review_full |
| 6 | Amazon Product Reviews | HuggingFace | https://huggingface.co/datasets/amazon_polarity |
| 7 | Women's Clothing Reviews | Kaggle | https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews |
| 8 | Fake & Real News | Kaggle | https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset |
| 9 | SMS Spam Collection | Kaggle | https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset |
| 10 | Email Spam/Ham | Kaggle | https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv |
| 11 | Stack Overflow Questions | Kaggle | https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate |
| 12 | DBpedia Ontology | HuggingFace | https://huggingface.co/datasets/dbpedia_14 |
| 13 | Medical Text (PubMed) | Kaggle | https://www.kaggle.com/datasets/chaitanyakck/medical-text |
| 14 | ArXiv Papers | Kaggle | https://www.kaggle.com/datasets/Cornell-University/arxiv |
| 15 | HuffPost News Category | Kaggle | https://www.kaggle.com/datasets/rmisra/news-category-dataset |
| 16 | Twitter Airline Sentiment | Kaggle | https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment |
| 17 | Reddit Self-posts | Kaggle | https://www.kaggle.com/datasets/mswarbrickjones/reddit-selfposts |
| 18 | Movie Plot Summaries (CMU) | Kaggle | https://www.kaggle.com/datasets/msafi04/movies-genre-dataset-cmu-movie-summary |
| 19 | Emotion (tweets) | HuggingFace | https://huggingface.co/datasets/dair-ai/emotion |

### C1. Prétraitement du dataset personnel

Avant d'appliquer les méthodes, préparez votre dataset :
- Chargement et inspection (taille, colonnes, types, valeurs manquantes)
- Nettoyage du texte : suppression de la ponctuation, mise en minuscules, suppression des stopwords
- Optionnel : lemmatisation ou stemming
- Vectorisation : TF-IDF (obligatoire), bag-of-words (optionnel)

### C2. Topic Modeling (LDA)

1. Appliquez LDA avec différentes valeurs de k (5, 10, 15, 20 thèmes)
2. Évaluez la qualité des thèmes (cohérence ou perplexité)
3. Affichez les 10 mots les plus représentatifs de chaque thème pour la meilleure valeur de k
4. Associez chaque document à son thème dominant
5. Visualisez la distribution des thèmes (diagramme en barres ou carte de chaleur thème × mots)

**Questions** : Quelle valeur de k vous semble optimale ? Les thèmes correspondent-ils aux catégories réelles du dataset ?

### C3. Classification supervisée

1. Utilisez les étiquettes (labels) du dataset comme classes cibles
2. Découpez en ensembles d'entraînement (80%) et de test (20%)
3. Entraînez au moins **deux classifieurs** parmi : Naive Bayes, Régression Logistique, SVM, Random Forest
4. Évaluez avec : accuracy, F1-score macro, matrice de confusion
5. Analysez les erreurs : quelles classes sont les plus confondues ?

**Questions** : Quel classifieur donne les meilleurs résultats ? Pourquoi ?


---

## Livrables

Vous devez remettre **deux livrables** par email avant la date limite (Mardi 02 Juin 00:00):

1. **Un dépôt GitHub public** contenant votre code et vos données
2. **Un rapport PDF** (10-15 pages)

### Structure du dépôt GitHub

Votre dépôt sera organisé comme suit :

```
TextMining_NOM_Prenom/
├── README.md            (description du projet, instructions d'exécution)
├── extraction.py        (complété)
├── exploration.py       (complété)
├── analyse.py           (complété)
├── requirements.txt     (liste des bibliothèques utilisées)
├── data/
│   ├── sara_wide.csv
│   ├── sara_long.csv
│   └── pairs/
│       ├── mbay_french.csv
│       └── ... (14 fichiers)
```

> **Note** : Ne publiez pas le fichier PDF `SaraLanguagesLexicon.pdf` ni les datasets Kaggle/HuggingFace sur GitHub. Indiquez dans le `README.md` comment les obtenir.

### Email de remise

Envoyez un email à donaldtitembaye@gmail.com avec :
- **Objet** : `[TextMining] NOM Prénom — Rendu`
- **Pièce jointe** : Votre rapport en PDF (`rapport_NOM_Prenom.pdf`)