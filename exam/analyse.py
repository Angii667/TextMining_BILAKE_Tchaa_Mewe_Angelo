"""
Partie C — Analyse avancée sur le dataset Medical Text.
Charge train.dat et test.dat, nettoie, vectorise, puis applique
Topic Modeling LDA, Classification supervisee et Clustering.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfVectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TRAIN_PATH = "data/train.dat"
TEST_PATH  = "data/test.dat"
FIGURES_DIR = Path("figures/partie_c")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class TextPreprocessor:

    def __init__(self, train_path: str = TRAIN_PATH, test_path: str = TEST_PATH):
        self.train_path = train_path
        self.test_path  = test_path
        self.df_train   = None
        self.df_test    = None
        self.texts      = None
        self.labels     = None
        self.test_texts = None

    def load(self):
        self.df_train = pd.read_csv(
            self.train_path, sep="\t", header=None,
            names=["condition_label", "medical_abstract"],
            encoding="utf-8", engine="c"
        )
        self.df_train["condition_label"] = (
            self.df_train["condition_label"].astype(int)
        )
        if self.test_path and Path(self.test_path).exists():
            self.df_test = pd.read_csv(
                self.test_path, sep="\t", header=None,
                names=["medical_abstract"],
                encoding="utf-8", engine="c"
            )

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", "", text)
        text = re.sub(r"\b\d+\s*(mg|ml|mm|cm|kg|g|μg|ui|mmol|mg/dl|h|yr|mo|wk|d|sec|min)\b",
                      " ", text)
        text = re.sub(r"\d+[/–-]\d+", " ", text)
        text = re.sub(r"[0-9]+\s*[xX×]\s*[0-9]+", " ", text)
        text = re.sub(r"[0-9]+\s*%", " ", text)
        text = re.sub(r'[0-9]+(?:\.[0-9]+)?', ' ', text)
        text = text.translate(str.maketrans(
            "", "", string.punctuation.replace("-", "")
        ))
        text = re.sub(r"-{2,}", " ", text)
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
        extra = {"p", "n", "et", "al", "vs", "ci", "ii", "iii", "iv", "vi",
                 "may", "also", "within", "without", "however", "thus",
                 "therefore", "among", "using", "including", "based",
                 "associated", "significant", "compared", "results",
                 "conclusion", "objective", "background", "method",
                 "purpose", "aim", "design", "setting", "population"}
        stop_words.update(extra)
        words = text.split()
        words = [w for w in words if w not in stop_words and len(w) > 1]
        return " ".join(words)

    def preprocess(self) -> tuple[list[str], list]:
        self.load()
        texts_raw = self.df_train["medical_abstract"].astype(str).tolist()
        self.labels = self.df_train["condition_label"].tolist()
        texts_clean = []
        for t in texts_raw:
            t = self.clean_text(t)
            t = self.remove_stopwords(t)
            texts_clean.append(t)
        self.texts = texts_clean
        if self.df_test is not None:
            self.test_texts = [
                self.remove_stopwords(self.clean_text(t))
                for t in self.df_test["medical_abstract"].astype(str).tolist()
            ]
        return texts_clean, self.labels

    def describe(self):
        n_docs = len(self.texts)
        n_classes = len(set(self.labels)) if self.labels else 0
        avg_len = np.mean([len(t.split()) for t in self.texts]) if self.texts else 0

        print(f"Documents      : {n_docs}")
        print(f"Classes        : {n_classes}")
        if self.labels:
            label_map = {1: "Digestif", 2: "Cardiovasculaire",
                         3: "Tumeurs", 4: "Nerveux", 5: "General"}
            dist = pd.Series(self.labels).value_counts().sort_index()
            print(f"Distribution   :")
            for cls, cnt in dist.items():
                name = label_map.get(cls, cls)
                print(f"  {cls} ({name:16s}) : {cnt:5d} ({cnt/n_docs*100:.1f}%)")
        print(f"Longueur moy.  : {avg_len:.1f} mots")
        if self.labels:
            for cls in sorted(set(self.labels)):
                ex = next(
                    (t[:120] for t, l in zip(self.texts, self.labels) if l == cls),
                    ""
                )
                if ex:
                    print(f"  Ex. [{cls}]: {ex}...")


class TopicModeler:

    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        self.best_model   = None
        self.best_k       = None
        self.count_vectorizer = None
        self.count_matrix     = None

    def fit_lda(self, n_topics: int, max_iter: int = 20) -> LatentDirichletAllocation:
        if self.count_matrix is None:
            self.count_vectorizer = CountVectorizer(
                max_features=2000, min_df=3, max_df=0.85,
                ngram_range=(1, 2)
            )
            self.count_matrix = self.count_vectorizer.fit_transform(
                self.preprocessor.texts
            )
        lda = LatentDirichletAllocation(
            n_components=n_topics, max_iter=max_iter,
            random_state=42, n_jobs=-1
        )
        lda.fit(self.count_matrix)
        return lda

    def select_best_k(self, k_values: list[int] = [5, 10, 15, 20]) -> int:
        perplexities = []
        models = {}

        for k in k_values:
            lda = self.fit_lda(k)
            perplexities.append(lda.perplexity(self.count_matrix))
            models[k] = lda
            print(f"  k={k:2d}  perplexite={perplexities[-1]:.2f}")

        best_k = k_values[np.argmin(perplexities)]
        self.best_k = best_k
        self.best_model = models[best_k]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_values, perplexities, marker="o", color="steelblue")
        ax.set_xlabel("Nombre de themes (k)")
        ax.set_ylabel("Perplexite")
        ax.set_title("Selection de k  Perplexite LDA")
        ax.scatter([best_k], [min(perplexities)], color="red", s=100, zorder=5,
                   label=f"k optimal = {best_k}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "lda_perplexite.png", dpi=150)
        plt.close()
        return best_k

    def display_topics(self, n_words: int = 12):
        terms = self.count_vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.best_model.components_):
            top_indices = topic.argsort()[:-n_words - 1:-1]
            top_terms = [terms[i] for i in top_indices]
            print(f"  Theme {topic_idx + 1:2d} : {', '.join(top_terms)}")

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
        ax.set_xlabel("Theme")
        ax.set_ylabel("Nombre de documents")
        ax.set_title("Distribution des documents par theme dominant")
        ax.set_xticks(range(1, self.best_k + 1))
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "lda_distribution_themes.png", dpi=150)
        plt.close()

    def run(self):
        print("\n=== C2. Topic Modeling ===")
        self.best_k = self.select_best_k()
        print(f"  Meilleur k : {self.best_k}")
        print("  Themes :")
        self.display_topics()
        df = self.assign_topics()
        print(f"  {len(df)} documents assignes a un theme")
        self.plot_topic_distribution()


class TextClassifier:

    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        self.results = {}

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        X = self.preprocessor.texts
        y = self.preprocessor.labels
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    def train_naive_bayes(self, X_train, y_train):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=8000, min_df=2, max_df=0.90,
                ngram_range=(1, 2), sublinear_tf=True
            )),
            ("clf", MultinomialNB(alpha=0.1)),
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    def train_logistic_regression(self, X_train, y_train):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=8000, min_df=2, max_df=0.90,
                ngram_range=(1, 2), sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                C=1.0, max_iter=2000, random_state=42, solver="lbfgs",
            )),
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    def train_svm(self, X_train, y_train):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=8000, min_df=2, max_df=0.90,
                ngram_range=(1, 2), sublinear_tf=True
            )),
            ("clf", LinearSVC(C=1.0, random_state=42, max_iter=3000)),
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    def train_random_forest(self, X_train, y_train):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=8000, min_df=2, max_df=0.90,
                ngram_range=(1, 2), sublinear_tf=True
            )),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=50, random_state=42,
                n_jobs=-1, min_samples_leaf=2
            )),
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    def evaluate(self, model, X_test, y_test, model_name: str):
        y_pred = model.predict(X_test)
        acc = (y_pred == y_test).mean()
        report = classification_report(y_test, y_pred, output_dict=True,
                                       zero_division=0)

        print(f"    Accuracy  : {acc:.4f}")
        print(f"    F1-macro  : {report['macro avg']['f1-score']:.4f}")
        print(f"    F1-weighted : {report['weighted avg']['f1-score']:.4f}")

        self.results[model_name] = {
            "accuracy": acc,
            "f1_macro": report["macro avg"]["f1-score"],
            "f1_weighted": report["weighted avg"]["f1-score"],
        }

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax, cmap="Blues", normalize="true"
        )
        ax.set_title(f"Matrice de confusion  {model_name}")
        plt.tight_layout()
        safe = model_name.replace(" ", "_").lower()
        plt.savefig(FIGURES_DIR / f"confusion_{safe}.png", dpi=150)
        plt.close()

    def compare_models(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(self.results))
        width = 0.25
        metrics = ["accuracy", "f1_macro", "f1_weighted"]
        colors = ["steelblue", "coral", "seagreen"]

        print("\n  Comparaison des modeles :")
        h = f"  {'Modele':25s} {'Accuracy':>10s} {'F1-macro':>10s} {'F1-weighted':>10s}"
        print(h)
        print("  " + "-" * 60)
        for i, (name, scores) in enumerate(self.results.items()):
            line = f"  {name:25s}"
            for metric in metrics:
                line += f" {scores[metric]:10.4f}"
            print(line)
            for j, metric in enumerate(metrics):
                ax.bar(x[i] + j * width, scores[metric], width,
                       color=colors[j], label=metric if i == 0 else "")

        ax.set_xticks(x + width)
        ax.set_xticklabels(self.results.keys(), rotation=15)
        ax.set_ylabel("Score")
        ax.set_title("Comparaison des classifieurs")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "comparaison_classifieurs.png", dpi=150)
        plt.close()

    def run(self):
        print("\n=== C3. Classification supervisee ===")
        X_train, X_test, y_train, y_test = self.split_data()
        print(f"  Train : {len(X_train)} docs  Test : {len(X_test)} docs")

        for name, trainer in [
            ("Naive Bayes",             self.train_naive_bayes),
            ("Regression Logistique",   self.train_logistic_regression),
            ("SVM Lineaire",            self.train_svm),
            ("Random Forest",           self.train_random_forest),
        ]:
            print(f"\n  Entrainement : {name}")
            model = trainer(X_train, y_train)
            self.evaluate(model, X_test, y_test, model_name=name)

        self.compare_models()


class TextClusterer:

    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        self.tfidf_matrix = None
        self.vectorizer   = None

    def _ensure_vectorized(self):
        if self.tfidf_matrix is None:
            self.vectorizer = TfidfVectorizer(
                max_features=5000, min_df=3, max_df=0.85,
                ngram_range=(1, 2), sublinear_tf=True
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(
                self.preprocessor.texts
            )

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
        ax1.set_title("Methode du coude (Elbow)")
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
        return km.fit_predict(self.tfidf_matrix)

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
        ax.set_title(f"Dendrogramme (echantillon de {dendro_size} docs)")
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

    def plot_clusters_2d(self, labels: np.ndarray, method: str = "pca",
                         title: str = ""):
        coords = self.reduce_dimensions(method)
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels,
                             cmap="tab10", alpha=0.6, s=10)
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
            if cluster_id == -1:
                continue
            mask = labels == cluster_id
            cluster_matrix = self.tfidf_matrix[mask]
            mean_scores = cluster_matrix.mean(axis=0).A1
            top_indices = mean_scores.argsort()[-n_words:][::-1]
            top_terms = [terms[i] for i in top_indices]
            print(f"  Cluster {cluster_id}: {', '.join(top_terms)}")

    def run(self):
        print("\n=== C4. Clustering ===")

        optimal_k = self.find_optimal_k()
        print(f"  k optimal : {optimal_k}")

        print("  K-Means...")
        km_labels = self.kmeans_clustering(optimal_k)
        self.plot_clusters_2d(km_labels, method="pca",
                              title="K-Means (PCA)")

        print("  Clustering hierarchique...")
        hc_labels = self.hierarchical_clustering(optimal_k)
        self.plot_clusters_2d(hc_labels, method="pca",
                              title="Hierarchique (PCA)")

        print("  Mots representatifs par cluster :")
        self.cluster_top_words(km_labels)


if __name__ == "__main__":

    prep = TextPreprocessor()
    prep.preprocess()
    prep.describe()

    tm = TopicModeler(prep)
    tm.run()

    clf = TextClassifier(prep)
    clf.run()

    clt = TextClusterer(prep)
    clt.run()
