"""
Partie C — Analyse avancée sur votre dataset personnel.

Complétez les trois classes :
  - TextPreprocessor  : prétraitement et vectorisation
  - TopicModeler      : modélisation thématique (LDA)
  - TextClassifier    : classification supervisée
  - TextClusterer     : clustering non supervisé

Remplacez DATASET_PATH par le chemin vers votre fichier de données,
et TEXT_COLUMN / LABEL_COLUMN par les noms des colonnes appropriées.

Consultez le sujet pour connaître votre dataset assigné.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             silhouette_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------
# CONFIGURATION — à adapter selon votre dataset
# ---------------------------------------------------------------

DATASET_PATH  = "votre_dataset.csv"   # chemin vers votre fichier
TEXT_COLUMN   = "text"                 # colonne contenant le texte
LABEL_COLUMN  = "label"                # colonne contenant la catégorie
FIGURES_DIR   = Path("figures/partie_c")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================
# 1. PRÉTRAITEMENT
# ==============================================================

class TextPreprocessor:
    """
    Charge, nettoie et vectorise le dataset personnel.
    """

    def __init__(self, dataset_path: str, text_col: str, label_col: str):
        """
        Args:
            dataset_path : Chemin vers le fichier de données (CSV ou chargé via HuggingFace)
            text_col     : Nom de la colonne texte
            label_col    : Nom de la colonne étiquette
        """
        self.text_col  = text_col
        self.label_col = label_col
        self.df        = None  # DataFrame brut
        self.texts     = None  # Série de textes nettoyés
        self.labels    = None  # Série d'étiquettes

    def load(self):
        """
        Charge les données depuis un fichier CSV ou via HuggingFace datasets.

        Pour les datasets HuggingFace (n° 2, 4, 5, 6, 12, 19), utilisez :
            from datasets import load_dataset
            dataset = load_dataset("nom_du_dataset")

        Conseil : inspectez les colonnes disponibles et la distribution des classes.
        """
        raise NotImplementedError("À implémenter")

    def clean_text(self, text: str) -> str:
        """
        Nettoie un texte brut :
        - Mise en minuscules
        - Suppression des URLs, chiffres, ponctuation
        - Suppression des espaces multiples

        Args:
            text : Texte brut à nettoyer

        Returns:
            Texte nettoyé
        """
        raise NotImplementedError("À implémenter")

    def remove_stopwords(self, text: str, language: str = "english") -> str:
        """
        Supprime les mots vides (stopwords) du texte.

        Utilisez nltk.corpus.stopwords ou spacy selon votre préférence.

        Args:
            text     : Texte déjà nettoyé
            language : Langue des stopwords ("english" ou "french")

        Returns:
            Texte sans stopwords
        """
        raise NotImplementedError("À implémenter")

    def preprocess(self) -> tuple[list[str], list]:
        """
        Applique le pipeline de nettoyage complet sur toutes les entrées.

        Ordre : load → clean_text → remove_stopwords

        Returns:
            (texts_clean, labels) : listes de textes nettoyés et d'étiquettes
        """
        raise NotImplementedError("À implémenter")

    def vectorize_tfidf(self, texts: list[str], max_features: int = 5000):
        """
        Vectorise les textes avec TF-IDF.

        Args:
            texts        : Liste de textes nettoyés
            max_features : Taille maximale du vocabulaire

        Returns:
            (matrix, vectorizer) : matrice TF-IDF sparse et l'objet TfidfVectorizer ajusté
        """
        raise NotImplementedError("À implémenter")

    def describe(self):
        """
        Affiche un résumé descriptif du dataset :
        - Nombre total de documents
        - Nombre de classes et distribution
        - Longueur moyenne des textes (en mots)
        - Exemples de textes par classe
        """
        raise NotImplementedError("À implémenter")


# ==============================================================
# 2. TOPIC MODELING
# ==============================================================

class TopicModeler:
    """
    Découvre des thèmes latents dans le corpus avec LDA.
    """

    def __init__(self, preprocessor: TextPreprocessor):
        """
        Args:
            preprocessor : Instance de TextPreprocessor déjà exécutée
        """
        self.preprocessor = preprocessor
        self.best_model   = None
        self.best_k       = None

    def fit_lda(self, n_topics: int, max_iter: int = 20) -> LatentDirichletAllocation:
        """
        Entraîne un modèle LDA avec n_topics thèmes.

        Utilisez CountVectorizer (bag-of-words) pour LDA,
        pas TF-IDF (LDA fonctionne mieux avec des comptages bruts).

        Args:
            n_topics : Nombre de thèmes
            max_iter : Nombre d'itérations

        Returns:
            Modèle LDA entraîné
        """
        raise NotImplementedError("À implémenter")

    def select_best_k(self, k_values: list[int] = [5, 10, 15, 20]) -> int:
        """
        Évalue LDA pour chaque valeur de k et sélectionne la meilleure.

        Critère : perplexité (plus basse = meilleur ajustement) ou
                  score de log-vraisemblance.

        Trace un graphique k vs perplexité.
        Sauvegarde : figures/partie_c/lda_perplexite.png

        Returns:
            Valeur optimale de k
        """
        raise NotImplementedError("À implémenter")

    def display_topics(self, n_words: int = 10):
        """
        Affiche les n_words mots les plus représentatifs de chaque thème
        pour le meilleur modèle.

        Format attendu :
            Thème 1 : mot1, mot2, mot3, ...
            Thème 2 : mot4, mot5, mot6, ...

        Args:
            n_words : Nombre de mots à afficher par thème
        """
        raise NotImplementedError("À implémenter")

    def assign_topics(self) -> pd.DataFrame:
        """
        Associe chaque document à son thème dominant.

        Returns:
            DataFrame avec colonnes : text, label (si dispo), dominant_topic, topic_score
        """
        raise NotImplementedError("À implémenter")

    def plot_topic_distribution(self):
        """
        Visualise la distribution des documents par thème dominant.

        Sauvegarde : figures/partie_c/lda_distribution_themes.png
        """
        raise NotImplementedError("À implémenter")

    def run(self):
        """Lance le pipeline Topic Modeling complet."""
        print("=== C2. Topic Modeling ===")
        self.best_k = self.select_best_k()
        print(f"  Meilleur k : {self.best_k}")
        self.display_topics()
        self.assign_topics()
        self.plot_topic_distribution()


# ==============================================================
# 3. CLASSIFICATION SUPERVISÉE
# ==============================================================

class TextClassifier:
    """
    Classifie les documents selon leurs étiquettes avec des modèles supervisés.
    """

    def __init__(self, preprocessor: TextPreprocessor):
        """
        Args:
            preprocessor : Instance de TextPreprocessor déjà exécutée
        """
        self.preprocessor = preprocessor
        self.results       = {}

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Divise les données en ensembles d'entraînement et de test.

        Args:
            test_size    : Proportion réservée au test (0.2 = 20%)
            random_state : Graine aléatoire pour la reproductibilité

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        raise NotImplementedError("À implémenter")

    def train_naive_bayes(self, X_train, y_train):
        """
        Entraîne un classifieur Naive Bayes (MultinomialNB).

        Conseil : utilisez un Pipeline sklearn pour enchaîner
        vectorisation + classification.

        Returns:
            Pipeline entraîné
        """
        raise NotImplementedError("À implémenter")

    def train_logistic_regression(self, X_train, y_train):
        """
        Entraîne une régression logistique (LogisticRegression).

        Conseil : max_iter=1000 pour assurer la convergence.

        Returns:
            Pipeline entraîné
        """
        raise NotImplementedError("À implémenter")

    def train_svm(self, X_train, y_train):
        """
        Entraîne un SVM linéaire (LinearSVC).

        Returns:
            Pipeline entraîné
        """
        raise NotImplementedError("À implémenter")

    def evaluate(self, model, X_test, y_test, model_name: str):
        """
        Évalue un modèle et affiche les métriques.

        Affiche :
        - Accuracy
        - F1-score macro et pondéré
        - Rapport de classification complet
        - Matrice de confusion (graphique)

        Sauvegarde : figures/partie_c/confusion_{model_name}.png

        Args:
            model      : Modèle entraîné
            X_test     : Données de test
            y_test     : Étiquettes réelles
            model_name : Nom du modèle (pour le titre et le nom de fichier)
        """
        raise NotImplementedError("À implémenter")

    def compare_models(self):
        """
        Affiche un tableau comparatif des performances de tous les modèles
        entraînés (accuracy, F1-macro, F1-pondéré).

        Sauvegarde : figures/partie_c/comparaison_classifieurs.png
        """
        raise NotImplementedError("À implémenter")

    def run(self):
        """Lance le pipeline Classification complet."""
        print("=== C3. Classification supervisée ===")
        X_train, X_test, y_train, y_test = self.split_data()
        print(f"  Train : {len(X_train)} docs | Test : {len(X_test)} docs")

        for name, trainer in [
            ("Naive Bayes",          self.train_naive_bayes),
            ("Régression Logistique", self.train_logistic_regression),
            ("SVM Linéaire",          self.train_svm),
        ]:
            print(f"\n  Entraînement : {name}")
            model = trainer(X_train, y_train)
            self.evaluate(model, X_test, y_test, model_name=name)

        self.compare_models()


# ==============================================================
# 4. CLUSTERING
# ==============================================================

class TextClusterer:
    """
    Regroupe les documents par similarité sans utiliser les étiquettes.
    """

    def __init__(self, preprocessor: TextPreprocessor):
        """
        Args:
            preprocessor : Instance de TextPreprocessor déjà exécutée
        """
        self.preprocessor = preprocessor

    def find_optimal_k(self, k_range: range = range(2, 11)) -> int:
        """
        Détermine le nombre optimal de clusters avec :
        - La méthode du coude (inertie de K-Means)
        - Le score de silhouette

        Trace les deux graphiques.
        Sauvegarde : figures/partie_c/kmeans_coude.png
                     figures/partie_c/kmeans_silhouette.png

        Returns:
            Valeur de k recommandée
        """
        raise NotImplementedError("À implémenter")

    def kmeans_clustering(self, n_clusters: int) -> np.ndarray:
        """
        Applique K-Means avec n_clusters clusters.

        Args:
            n_clusters : Nombre de clusters

        Returns:
            Tableau des étiquettes de clusters pour chaque document
        """
        raise NotImplementedError("À implémenter")

    def hierarchical_clustering(self, n_clusters: int):
        """
        Applique le clustering hiérarchique agglomératif.

        Trace le dendrogramme (sur un sous-échantillon si le corpus est large).
        Sauvegarde : figures/partie_c/dendrogramme.png

        Args:
            n_clusters : Nombre de clusters finaux

        Returns:
            Tableau des étiquettes de clusters
        """
        raise NotImplementedError("À implémenter")

    def reduce_dimensions(self, method: str = "pca") -> np.ndarray:
        """
        Réduit les données en 2 dimensions pour la visualisation.

        Args:
            method : "pca" (rapide) ou "tsne" (plus précis, plus lent)

        Returns:
            Tableau numpy de forme (n_docs, 2)
        """
        raise NotImplementedError("À implémenter")

    def plot_clusters_2d(self, labels: np.ndarray, method: str = "pca", title: str = ""):
        """
        Visualise les clusters en 2D avec les couleurs des clusters.
        Si les vraies étiquettes sont disponibles, les superpose pour comparaison.

        Sauvegarde : figures/partie_c/clusters_2d_{method}.png

        Args:
            labels : Étiquettes de clusters
            method : Méthode de réduction utilisée ("pca" ou "tsne")
            title  : Titre du graphique
        """
        raise NotImplementedError("À implémenter")

    def cluster_top_words(self, labels: np.ndarray, n_words: int = 10):
        """
        Pour chaque cluster, affiche les mots les plus fréquents.
        Cela permet d'interpréter ce que représente chaque cluster.

        Args:
            labels  : Étiquettes de clusters
            n_words : Nombre de mots à afficher par cluster
        """
        raise NotImplementedError("À implémenter")

    def run(self):
        """Lance le pipeline Clustering complet."""
        print("=== C4. Clustering ===")

        optimal_k = self.find_optimal_k()
        print(f"  k optimal : {optimal_k}")

        print("  K-Means...")
        km_labels = self.kmeans_clustering(optimal_k)
        self.plot_clusters_2d(km_labels, method="pca", title="K-Means (PCA)")

        print("  Clustering hiérarchique...")
        hc_labels = self.hierarchical_clustering(optimal_k)
        self.plot_clusters_2d(hc_labels, method="pca", title="Hiérarchique (PCA)")

        print("  Mots représentatifs par cluster :")
        self.cluster_top_words(km_labels)


# ==============================================================
# POINT D'ENTRÉE
# ==============================================================

if __name__ == "__main__":

    # 1. Prétraitement
    prep = TextPreprocessor(
        dataset_path=DATASET_PATH,
        text_col=TEXT_COLUMN,
        label_col=LABEL_COLUMN,
    )
    prep.preprocess()
    prep.describe()

    # 2. Topic Modeling
    tm = TopicModeler(prep)
    tm.run()

    # 3. Classification
    clf = TextClassifier(prep)
    clf.run()

    # 4. Clustering
    clt = TextClusterer(prep)
    clt.run()
