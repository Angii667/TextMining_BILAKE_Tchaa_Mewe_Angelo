"""
Partie C — Analyse avancée sur le dataset Medical Text.

Pipeline NLP complet en 4 étapes :
  1. Prétraitement (TextPreprocessor)   : chargement, nettoyage, stopwords
  2. Topic Modeling (TopicModeler)       : LDA, sélection de k, visualisation
  3. Classification (TextClassifier)     : 4 modèles supervisés + comparaison
  4. Clustering (TextClusterer)          : K-Means, hiérarchique, interprétation

Sources : data/train.dat (label + texte), data/test.dat (texte seul)
Format train.dat : tab-separé, label (1-5) en première colonne
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
    """
    Charge et nettoie les données médicales.

    Pipeline :
      load() → clean_text() → remove_stopwords() → preprocess()
    """

    def __init__(self, train_path: str = TRAIN_PATH, test_path: str = TEST_PATH):
        """
        Args:
            train_path : Chemin vers train.dat (tab-separé, label + texte)
            test_path  : Chemin vers test.dat (texte seul, optionnel)
        """
        self.train_path = train_path
        self.test_path  = test_path
        self.df_train   = None
        self.df_test    = None
        self.texts      = None
        self.labels     = None
        self.test_texts = None

    def load(self):
        """
        Charge les fichiers .dat (tab-separés, sans en-tête).

        train.dat : label (int) + medical_abstract (str)
        test.dat  : medical_abstract (str)
        """
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
        """
        Nettoie un texte médical brut.

        Opérations :
          - Mise en minuscules
          - Suppression des balises HTML/XML
          - Suppression des URLs, emails
          - Suppression des mesures médicales (10 mg, 5 ml, etc.)
          - Suppression des fractions, pourcentages, nombres
          - Suppression de la ponctuation (sauf traits d'union simples)
          - Normalisation des espaces

        Args:
            text : Texte brut à nettoyer

        Returns:
            Texte nettoyé
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        # Balises HTML/XML
        text = re.sub(r"<[^>]+>", "", text)
        # URLs et emails
        text = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", "", text)
        # Mesures médicales (dosages, unités)
        text = re.sub(r"\b\d+\s*(mg|ml|mm|cm|kg|g|μg|ui|mmol|mg/dl|h|yr|mo|wk|d|sec|min)\b",
                      " ", text)
        # Fractions et plages numériques
        text = re.sub(r"\d+[/–-]\d+", " ", text)
        # Dimensions (3x4, 2×3)
        text = re.sub(r"[0-9]+\s*[xX×]\s*[0-9]+", " ", text)
        # Pourcentages
        text = re.sub(r"[0-9]+\s*%", " ", text)
        # Nombres (entiers et décimaux)
        text = re.sub(r'[0-9]+(?:\.[0-9]+)?', ' ', text)
        # Ponctuation (conserve le trait d'union simple)
        text = text.translate(str.maketrans(
            "", "", string.punctuation.replace("-", "")
        ))
        # Traits d'union multiples (garder simple)
        text = re.sub(r"-{2,}", " ", text)
        # Espaces multiples
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def remove_stopwords(self, text: str, language: str = "english") -> str:
        """
        Supprime les mots vides (stopwords) du texte.

        Ajoute des stopwords spécifiques au domaine médical
        (mots méthodologiques : "results", "conclusion", etc.).

        Args:
            text     : Texte déjà nettoyé
            language : Langue des stopwords ("english" par défaut)

        Returns:
            Texte sans stopwords
        """
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
        # Stopwords supplémentaires spécifiques au domaine médical
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
        """
        Applique le pipeline complet de nettoyage sur tout le corpus.

        Ordre : load → clean_text → remove_stopwords

        Returns:
            (texts_clean, labels) : listes des textes nettoyés et des étiquettes
        """
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
        """
        Affiche un résumé descriptif du dataset :
          - Nombre de documents, classes, distribution
          - Longueur moyenne des textes
          - Exemples de textes par classe
        """
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
    """
    Modélisation thématique avec LDA (Latent Dirichlet Allocation).

    Pipeline :
      fit_lda() → select_best_k() → display_topics() → assign_topics()
    """

    def __init__(self, preprocessor: TextPreprocessor):
        """
        Args:
            preprocessor : Instance de TextPreprocessor déjà exécutée
        """
        self.preprocessor = preprocessor
        self.best_model   = None
        self.best_k       = None
        self.count_vectorizer = None
        self.count_matrix     = None

    def fit_lda(self, n_topics: int, max_iter: int = 20) -> LatentDirichletAllocation:
        """
        Entraîne un modèle LDA avec n_topics thèmes.

        Utilise CountVectorizer (bag-of-words) car LDA fonctionne
        mieux avec des comptages bruts qu'avec TF-IDF.

        Args:
            n_topics : Nombre de thèmes à extraire
            max_iter : Nombre d'itérations pour la convergence

        Returns:
            Modèle LDA entraîné
        """
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
        """
        Évalue LDA pour différentes valeurs de k et sélectionne l'optimum.

        Critère : perplexité (plus basse = meilleur ajustement).
        Affiche et sauvegarde le graphique k vs perplexité.

        Sauvegarde : figures/partie_c/lda_perplexite.png

        Args:
            k_values : Liste des valeurs de k à tester

        Returns:
            Valeur optimale de k (celle qui minimise la perplexité)
        """
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
        """
        Affiche les n mots les plus représentatifs de chaque thème.

        Format :
          Theme 1 : mot1, mot2, mot3, ...

        Args:
            n_words : Nombre de mots à afficher par thème
        """
        terms = self.count_vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.best_model.components_):
            top_indices = topic.argsort()[:-n_words - 1:-1]
            top_terms = [terms[i] for i in top_indices]
            print(f"  Theme {topic_idx + 1:2d} : {', '.join(top_terms)}")

    def assign_topics(self) -> pd.DataFrame:
        """
        Associe chaque document à son thème dominant.

        Returns:
            DataFrame avec colonnes :
              - text           : texte original nettoyé
              - dominant_topic : index du thème dominant
              - topic_score    : probabilité du thème dominant
              - label          : classe réelle (si disponible)
        """
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
        """
        Visualise la distribution des documents par thème dominant.

        Sauvegarde : figures/partie_c/lda_distribution_themes.png
        """
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
        """
        Lance le pipeline Topic Modeling complet :
          sélection de k → affichage des thèmes → assignment → visualisation
        """
        print("\n=== C2. Topic Modeling ===")
        self.best_k = self.select_best_k()
        print(f"  Meilleur k : {self.best_k}")
        print("  Themes :")
        self.display_topics()
        df = self.assign_topics()
        print(f"  {len(df)} documents assignes a un theme")
        self.plot_topic_distribution()


class TextClassifier:
    """
    Classification supervisée avec 4 modèles :
      - Naive Bayes (MultinomialNB)
      - Régression Logistique (LogisticRegression)
      - SVM Linéaire (LinearSVC)
      - Random Forest (RandomForestClassifier)
    """

    def __init__(self, preprocessor: TextPreprocessor):
        """
        Args:
            preprocessor : Instance de TextPreprocessor déjà exécutée
        """
        self.preprocessor = preprocessor
        self.results = {}

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Divise les données en train/test avec stratification.

        Args:
            test_size    : Proportion réservée au test (défaut: 20%)
            random_state : Graine aléatoire pour la reproductibilité

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        X = self.preprocessor.texts
        y = self.preprocessor.labels
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    def train_naive_bayes(self, X_train, y_train):
        """
        Entraîne un classifieur Naive Bayes multinomial.

        Pipeline : TF-IDF (unigrammes + bigrammes) → MultinomialNB

        Returns:
            Pipeline entraîné
        """
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
        """
        Entraîne une régression logistique multinomiale.

        Pipeline : TF-IDF → LogisticRegression (solver lbfgs)

        Returns:
            Pipeline entraîné
        """
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
        """
        Entraîne un SVM linéaire.

        Pipeline : TF-IDF → LinearSVC

        Returns:
            Pipeline entraîné
        """
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
        """
        Entraîne une forêt aléatoire.

        Pipeline : TF-IDF → RandomForest (200 arbres, profondeur max 50)

        Returns:
            Pipeline entraîné
        """
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
        """
        Évalue un modèle et affiche les métriques.

        Affiche : accuracy, F1-macro, F1-pondéré
        Sauvegarde : matrice de confusion normalisée

        Sauvegarde : figures/partie_c/confusion_{model_name}.png

        Args:
            model      : Modèle entraîné (Pipeline sklearn)
            X_test     : Textes de test
            y_test     : Étiquettes réelles
            model_name : Nom du modèle (pour l'affichage et le fichier)
        """
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
        """
        Compare les performances de tous les modèles entraînés.

        Affiche un tableau comparatif (accuracy, F1-macro, F1-pondéré).
        Sauvegarde un graphique à barres groupées.

        Sauvegarde : figures/partie_c/comparaison_classifieurs.png
        """
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
        """
        Lance le pipeline Classification complet :
          split → entraîne 4 modèles → évalue → compare
        """
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
    """
    Clustering non supervisé des documents médicaux.

    Méthodes :
      - K-Means : partitionnement en k clusters
      - Hiérarchique : clustering agglomératif avec dendrogramme
      - Réduction de dimension : PCA ou t-SNE pour visualisation 2D
    """

    def __init__(self, preprocessor: TextPreprocessor):
        """
        Args:
            preprocessor : Instance de TextPreprocessor déjà exécutée
        """
        self.preprocessor = preprocessor
        self.tfidf_matrix = None
        self.vectorizer   = None

    def _ensure_vectorized(self):
        """
        Vectorise les textes en TF-IDF si ce n'est pas déjà fait.

        Utilise : unigrammes + bigrammes, 5000 features max,
        min_df=3, max_df=0.85, sublinear_tf=True
        """
        if self.tfidf_matrix is None:
            self.vectorizer = TfidfVectorizer(
                max_features=5000, min_df=3, max_df=0.85,
                ngram_range=(1, 2), sublinear_tf=True
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(
                self.preprocessor.texts
            )

    def find_optimal_k(self, k_range: range = range(2, 11)) -> int:
        """
        Détermine le nombre optimal de clusters.

        Utilise deux critères :
          - Méthode du coude (inertie intra-cluster)
          - Score de silhouette (cohérence des clusters)

        Sauvegarde : figures/partie_c/kmeans_coude.png
                     figures/partie_c/kmeans_silhouette.png

        Args:
            k_range : Plage de valeurs de k à tester

        Returns:
            Valeur optimale de k (maximise le silhouette score)
        """
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
        """
        Applique l'algorithme K-Means.

        Args:
            n_clusters : Nombre de clusters

        Returns:
            Tableau des étiquettes de cluster pour chaque document
        """
        self._ensure_vectorized()
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        return km.fit_predict(self.tfidf_matrix)

    def hierarchical_clustering(self, n_clusters: int):
        """
        Applique le clustering hiérarchique agglomératif (Ward).

        Utilise un sous-échantillon de 2000 documents max pour
        le clustering et 500 pour le dendrogramme (performance).

        Sauvegarde : figures/partie_c/dendrogramme.png

        Args:
            n_clusters : Nombre de clusters finaux

        Returns:
            Tableau des étiquettes de cluster (taille complète,
            -1 pour les documents non échantillonnés)
        """
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
        """
        Réduit la matrice TF-IDF en 2 dimensions pour visualisation.

        Args:
            method : "pca" (rapide, linéaire) ou "tsne" (précis, lent)

        Returns:
            Coordonnées 2D des documents (n_docs, 2)
        """
        self._ensure_vectorized()
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            reducer = PCA(n_components=2, random_state=42)
        return reducer.fit_transform(self.tfidf_matrix.toarray())

    def plot_clusters_2d(self, labels: np.ndarray, method: str = "pca",
                         title: str = ""):
        """
        Visualise les clusters en 2D après réduction de dimension.

        Sauvegarde : figures/partie_c/clusters_2d_{method}.png

        Args:
            labels : Étiquettes de cluster pour chaque document
            method : Méthode de réduction ("pca" ou "tsne")
            title  : Titre du graphique
        """
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
        """
        Pour chaque cluster, affiche les mots les plus représentatifs
        (score TF-IDF moyen le plus élevé).

        Permet d'interpréter sémantiquement chaque cluster.

        Args:
            labels  : Étiquettes de cluster
            n_words : Nombre de mots à afficher par cluster
        """
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
        """
        Lance le pipeline Clustering complet :
          k optimal → K-Means + visualisation → hiérarchique + dendrogramme
          → mots représentatifs par cluster
        """
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

    # 1. Prétraitement
    prep = TextPreprocessor()
    prep.preprocess()
    prep.describe()

    # 2. Topic Modeling (LDA)
    tm = TopicModeler(prep)
    tm.run()

    # 3. Classification supervisée (4 modèles)
    clf = TextClassifier(prep)
    clf.run()

    # 4. Clustering (K-Means + hiérarchique)
    clt = TextClusterer(prep)
    clt.run()
