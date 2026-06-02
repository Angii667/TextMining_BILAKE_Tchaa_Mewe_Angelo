"""
Partie C — Analyse avancée sur votre dataset personnel.

4 classes : TextPreprocessor, TopicModeler, TextClassifier, TextClusterer

Configurez DATASET_PATH, TEXT_COLUMN, LABEL_COLUMN selon votre dataset.
"""

import re
import string
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------
# CONFIGURATION — à adapter selon votre dataset
# ---------------------------------------------------------------

DATASET_PATH  = "data/medical_text.csv"
TEXT_COLUMN   = "medical_abstract"
LABEL_COLUMN  = "condition_label"
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
        self.dataset_path = dataset_path
        self.text_col  = text_col
        self.label_col = label_col
        self.df        = None
        self.texts     = None
        self.labels    = None

    def load(self):
        if self.dataset_path.endswith(".csv"):
            self.df = pd.read_csv(self.dataset_path)
        else:
            try:
                from datasets import load_dataset
                dataset = load_dataset(self.dataset_path)
                if "train" in dataset:
                    self.df = dataset["train"].to_pandas()
                else:
                    self.df = dataset[list(dataset.keys())[0]].to_pandas()
            except ImportError:
                raise ImportError("datasets library not installed. Use a CSV file instead.")
            except Exception as e:
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path} ({e})")

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def remove_stopwords(self, text: str, language: str = "english") -> str:
        try:
            from nltk.corpus import stopwords
        except ImportError:
            return text
        try:
            stop_words = set(stopwords.words(language))
        except LookupError:
            import nltk
            nltk.download("stopwords", quiet=True)
            stop_words = set(stopwords.words(language))
        words = text.split()
        words = [w for w in words if w not in stop_words]
        return " ".join(words)

    def preprocess(self) -> tuple[list[str], list]:
        self.load()
        texts_raw = self.df[self.text_col].astype(str).tolist()
        labels = self.df[self.label_col].tolist() if self.label_col in self.df.columns else []

        texts_clean = []
        for t in texts_raw:
            t = self.clean_text(t)
            t = self.remove_stopwords(t)
            texts_clean.append(t)

        self.texts = texts_clean
        self.labels = labels
        return texts_clean, labels

    def vectorize_tfidf(self, texts: list[str], max_features: int = 5000):
        vectorizer = TfidfVectorizer(max_features=max_features)
        matrix = vectorizer.fit_transform(texts)
        return matrix, vectorizer

    def describe(self):
        n_docs = len(self.texts)
        n_classes = len(set(self.labels)) if self.labels else 0
        avg_len = np.mean([len(t.split()) for t in self.texts]) if self.texts else 0

        print(f"Documents      : {n_docs}")
        print(f"Classes        : {n_classes}")
        if self.labels:
            dist = pd.Series(self.labels).value_counts()
            print(f"Distribution   :")
            for cls, cnt in dist.items():
                print(f"  {cls}: {cnt} ({cnt/n_docs*100:.1f}%)")
        print(f"Longueur moy.  : {avg_len:.1f} mots")
        if self.labels:
            for cls in sorted(set(self.labels)):
                ex = next((t[:120] for t, l in zip(self.texts, self.labels) if l == cls), "")
                if ex:
                    print(f"  Ex. [{cls}]: {ex}...")


# ==============================================================
# 2. TOPIC MODELING
# ==============================================================

class TopicModeler:
    """
    Découvre des thèmes latents dans le corpus avec LDA.
    """

    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        self.best_model   = None
        self.best_k       = None
        self.count_vectorizer = None
        self.count_matrix     = None

    def fit_lda(self, n_topics: int, max_iter: int = 15) -> LatentDirichletAllocation:
        if self.count_matrix is None:
            self.count_vectorizer = CountVectorizer(max_features=1500)
            self.count_matrix = self.count_vectorizer.fit_transform(self.preprocessor.texts)
        lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter,
                                        random_state=42)
        lda.fit(self.count_matrix)
        return lda

    def select_best_k(self, k_values: list[int] = [5, 10, 15, 20]) -> int:
        perplexities = []
        models = {}

        for k in k_values:
            lda = self.fit_lda(k)
            perplexities.append(lda.perplexity(self.count_matrix))
            models[k] = lda
            print(f"  k={k:2d}  perplexité={perplexities[-1]:.2f}")

        best_k = k_values[np.argmin(perplexities)]
        self.best_k = best_k
        self.best_model = models[best_k]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_values, perplexities, marker="o", color="steelblue")
        ax.set_xlabel("Nombre de thèmes (k)")
        ax.set_ylabel("Perplexité")
        ax.set_title("Sélection de k — Perplexité LDA")
        ax.scatter([best_k], [min(perplexities)], color="red", s=100, zorder=5,
                   label=f"k optimal = {best_k}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "lda_perplexite.png", dpi=150)
        plt.close()

        return best_k

    def display_topics(self, n_words: int = 10):
        terms = self.count_vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.best_model.components_):
            top_indices = topic.argsort()[:-n_words - 1:-1]
            top_terms = [terms[i] for i in top_indices]
            print(f"  Thème {topic_idx + 1:2d} : {', '.join(top_terms)}")

    def assign_topics(self) -> pd.DataFrame:
        topic_dist = self.best_model.transform(self.count_matrix)
        dominant = topic_dist.argmax(axis=1)
        scores = topic_dist.max(axis=1)

        df = pd.DataFrame({
            "text": self.preprocessor.texts,
            "dominant_topic": dominant,
            "topic_score": scores,
        })
        if self.preprocessor.labels:
            df["label"] = self.preprocessor.labels

        return df

    def plot_topic_distribution(self):
        topic_dist = self.best_model.transform(self.count_matrix)
        dominant = topic_dist.argmax(axis=1)

        fig, ax = plt.subplots(figsize=(10, 5))
        counts = pd.Series(dominant).value_counts().sort_index()
        ax.bar(counts.index + 1, counts.values, color="steelblue", edgecolor="black")
        ax.set_xlabel("Thème")
        ax.set_ylabel("Nombre de documents")
        ax.set_title("Distribution des documents par thème dominant")
        ax.set_xticks(range(1, self.best_k + 1))
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "lda_distribution_themes.png", dpi=150)
        plt.close()

    def run(self):
        print("=== C2. Topic Modeling ===")
        self.best_k = self.select_best_k()
        print(f"  Meilleur k : {self.best_k}")
        print("  Thèmes :")
        self.display_topics()
        df = self.assign_topics()
        print(f"  {len(df)} documents assignés à un thème")
        self.plot_topic_distribution()


# ==============================================================
# 3. CLASSIFICATION SUPERVISÉE
# ==============================================================

class TextClassifier:
    """
    Classifie les documents selon leurs étiquettes avec des modèles supervisés.
    """

    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        self.results = {}

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        X = self.preprocessor.texts
        y = self.preprocessor.labels
        return train_test_split(X, y, test_size=test_size, random_state=random_state,
                                stratify=y)

    def train_naive_bayes(self, X_train, y_train):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", MultinomialNB()),
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    def train_logistic_regression(self, X_train, y_train):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    def train_svm(self, X_train, y_train):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", LinearSVC(random_state=42, max_iter=2000)),
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    def evaluate(self, model, X_test, y_test, model_name: str):
        y_pred = model.predict(X_test)
        acc = (y_pred == y_test).mean()
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        print(f"    Accuracy  : {acc:.4f}")
        print(f"    F1-macro  : {report['macro avg']['f1-score']:.4f}")
        print(f"    F1-weighted : {report['weighted avg']['f1-score']:.4f}")

        self.results[model_name] = {
            "accuracy": acc,
            "f1_macro": report["macro avg"]["f1-score"],
            "f1_weighted": report["weighted avg"]["f1-score"],
        }

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax,
                                                 cmap="Blues", normalize="true")
        ax.set_title(f"Matrice de confusion — {model_name}")
        plt.tight_layout()
        safe_name = model_name.replace(" ", "_").lower()
        plt.savefig(FIGURES_DIR / f"confusion_{safe_name}.png", dpi=150)
        plt.close()

    def compare_models(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(self.results))
        width = 0.25
        metrics = ["accuracy", "f1_macro", "f1_weighted"]
        colors = ["steelblue", "coral", "seagreen"]

        print("\n  Comparaison des modèles :")
        print(f"  {'Modèle':25s} {'Accuracy':>10s} {'F1-macro':>10s} {'F1-weighted':>10s}")
        print("  " + "-" * 60)
        for i, (name, scores) in enumerate(self.results.items()):
            print(f"  {name:25s} {scores['accuracy']:10.4f} {scores['f1_macro']:10.4f} {scores['f1_weighted']:10.4f}")
            for j, metric in enumerate(metrics):
                ax.bar(x[i] + j * width, scores[metric], width, color=colors[j],
                       label=metric if i == 0 else "")

        ax.set_xticks(x + width)
        ax.set_xticklabels(self.results.keys(), rotation=15)
        ax.set_ylabel("Score")
        ax.set_title("Comparaison des classifieurs")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "comparaison_classifieurs.png", dpi=150)
        plt.close()

    def run(self):
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
        self.preprocessor = preprocessor
        self.tfidf_matrix = None
        self.vectorizer   = None

    def _ensure_vectorized(self):
        if self.tfidf_matrix is None:
            self.vectorizer = TfidfVectorizer(max_features=5000)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.preprocessor.texts)

    def find_optimal_k(self, k_range: range = range(2, 11)) -> int:
        self._ensure_vectorized()
        inertias = []
        silhouettes = []

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(self.tfidf_matrix)
            inertias.append(km.inertia_)
            sil = silhouette_score(self.tfidf_matrix, labels)
            silhouettes.append(sil)
            print(f"  k={k:2d}  inertia={km.inertia_:.0f}  silhouette={sil:.4f}")

        optimal_k = k_range[np.argmax(silhouettes)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(list(k_range), inertias, marker="o", color="steelblue")
        ax1.set_xlabel("k")
        ax1.set_ylabel("Inertie")
        ax1.set_title("Méthode du coude (Elbow)")
        ax1.scatter([optimal_k], [inertias[list(k_range).index(optimal_k)]],
                    color="red", s=100, zorder=5)
        plt.savefig(FIGURES_DIR / "kmeans_coude.png", dpi=150)

        ax2.plot(list(k_range), silhouettes, marker="o", color="coral")
        ax2.set_xlabel("k")
        ax2.set_ylabel("Score de silhouette")
        ax2.set_title("Score de silhouette")
        ax2.scatter([optimal_k], [max(silhouettes)], color="red", s=100, zorder=5)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "kmeans_silhouette.png", dpi=150)
        plt.close()

        return optimal_k

    def kmeans_clustering(self, n_clusters: int) -> np.ndarray:
        self._ensure_vectorized()
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = km.fit_predict(self.tfidf_matrix)
        return labels

    def hierarchical_clustering(self, n_clusters: int):
        self._ensure_vectorized()
        sample_size = min(2000, self.tfidf_matrix.shape[0])
        idx = np.random.choice(self.tfidf_matrix.shape[0], sample_size, replace=False)
        hc = AgglomerativeClustering(n_clusters=n_clusters)
        labels = np.full(self.tfidf_matrix.shape[0], -1)
        labels[idx] = hc.fit_predict(self.tfidf_matrix[idx].toarray())

        from scipy.cluster.hierarchy import dendrogram, linkage
        dendro_size = min(500, sample_size)
        idx2 = np.random.choice(idx, dendro_size, replace=False)
        Z = linkage(self.tfidf_matrix[idx2].toarray(), method="ward")
        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.7 * Z[-1, 2])
        ax.set_title(f"Dendrogramme (échantillon de {dendro_size} docs)")
        ax.set_xlabel("Documents")
        ax.set_ylabel("Distance")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "dendrogramme.png", dpi=150)
        plt.close()

        return labels

    def reduce_dimensions(self, method: str = "pca") -> np.ndarray:
        self._ensure_vectorized()
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            reducer = PCA(n_components=2, random_state=42)
        return reducer.fit_transform(self.tfidf_matrix.toarray())

    def plot_clusters_2d(self, labels: np.ndarray, method: str = "pca", title: str = ""):
        coords = self.reduce_dimensions(method)
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10",
                             alpha=0.6, s=10)
        ax.set_title(title)
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"clusters_2d_{method}.png", dpi=150)
        plt.close()

    def cluster_top_words(self, labels: np.ndarray, n_words: int = 10):
        terms = self.vectorizer.get_feature_names_out()
        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id
            cluster_matrix = self.tfidf_matrix[mask]
            mean_scores = cluster_matrix.mean(axis=0).A1
            top_indices = mean_scores.argsort()[-n_words:][::-1]
            top_terms = [terms[i] for i in top_indices]
            print(f"  Cluster {cluster_id}: {', '.join(top_terms)}")

    def run(self):
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

    prep = TextPreprocessor(
        dataset_path=DATASET_PATH,
        text_col=TEXT_COLUMN,
        label_col=LABEL_COLUMN,
    )
    prep.preprocess()
    prep.describe()

    tm = TopicModeler(prep)
    tm.run()

    clf = TextClassifier(prep)
    clf.run()

    clt = TextClusterer(prep)
    clt.run()
