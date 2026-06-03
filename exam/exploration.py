"""
Partie B — Exploration statistique du lexique Sara.

Ce module analyse le lexique multilingue Sara (14 dialectes tchadiens + français).
Il utilise le format large (sara_wide.csv) comme source principale pour les stats
et le format long (sara_long.csv) pour les agrégations par dialecte.

Produit 5 graphiques dans figures/ :
  - paires_par_dialecte.png   : barres horizontales du nombre de paires par dialecte
  - heatmap_couverture.png    : matrice binaire de présence des termes par dialecte
  - longueur_mots_boxplot.png : distribution des longueurs de mots
  - wordcloud_Mb.png          : nuage de mots Mbay
  - wordcloud_Ngb.png         : nuage de mots Ngambay
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class SaraExplorer:
    """
    Analyse statistique et visualisation du lexique multilingue Sara.

    Fonctionnalités :
      - B1. Statistiques de base (termes, paires, couverture)
      - B2. Analyse du vocabulaire (fréquence, longueur des mots)
      - B3. Visualisations (barres, heatmap, boîtes, nuages de mots)
    """

    # Dictionnaire de correspondance : code dialecte → nom complet
    LANG_NAMES = {
        "Beb": "Bebote",   "Bd": "Bediondo", "Db": "Daba",
        "Gor": "Gor",      "Gu": "Gulay",    "KbN": "Kaba_Na",
        "Kbb": "Kaba",     "Lk": "Laka",     "Mb": "Mbay",
        "Mo":  "Mango",    "Nar": "Nar",      "Ngb": "Ngambay",
        "Sr":  "Sar",      "NgT": "Ngam",
    }

    def __init__(self, wide_csv: str, long_csv: str, figures_dir: str = "figures/"):
        """
        Initialise l'explorateur avec les deux fichiers CSV.

        Args:
            wide_csv    : Chemin vers sara_wide.csv (1 ligne = 1 terme français,
                         1 colonne par dialecte)
            long_csv    : Chemin vers sara_long.csv (1 ligne = 1 paire de traduction)
            figures_dir : Dossier de sauvegarde des graphiques
        """
        self.df_wide = pd.read_csv(wide_csv, encoding="utf-8-sig")
        self.df_long = pd.read_csv(long_csv, encoding="utf-8-sig")
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        # Liste des colonnes correspondant aux dialectes (tout sauf french_term)
        self.lang_cols = [c for c in self.df_wide.columns if c != "french_term"]

    # ------------------------------------------------------------------
    # B1. Statistiques de base
    # ------------------------------------------------------------------

    def basic_stats(self) -> dict:
        """
        B1 — Calcule et affiche les statistiques descriptives de base.

        - Nombre de termes français uniques
        - Nombre total de paires de traduction
        - Répartition des paires par dialecte
        - Taux de couverture (pourcentage de termes traduits) par dialecte

        Returns:
            Dictionnaire contenant n_terms, n_pairs, pairs_per_dialect, coverage
        """
        n_terms = self.df_wide["french_term"].nunique()
        n_pairs = len(self.df_long)
        pairs_per_dialect = self.df_long["target_lang"].value_counts()
        total_terms = n_terms
        coverage = {}
        for code, name in self.LANG_NAMES.items():
            n_translated = self.df_wide[code].notna() & (self.df_wide[code] != "")
            coverage[name] = round(n_translated.sum() / total_terms * 100, 2)

        print(f"Termes français uniques : {n_terms}")
        print(f"Paires de traduction    : {n_pairs}")
        print(f"\nPaires par dialecte :")
        for lang, count in pairs_per_dialect.items():
            print(f"  {lang:12s} : {count}")
        print(f"\nTaux de couverture par dialecte :")
        for lang, pct in coverage.items():
            print(f"  {lang:12s} : {pct}%")

        return {
            "n_terms": n_terms,
            "n_pairs": n_pairs,
            "pairs_per_dialect": pairs_per_dialect.to_dict(),
            "coverage": coverage,
        }

    # ------------------------------------------------------------------
    # B2. Analyse du vocabulaire
    # ------------------------------------------------------------------

    def coverage_analysis(self) -> pd.DataFrame:
        """
        B2 — Analyse la couverture des termes à travers les dialectes.

        - Top 20 des termes présents dans le plus de dialectes
        - Bottom 20 des termes les moins représentés
        - Termes présents dans TOUS les dialectes

        Returns:
            DataFrame avec pour chaque terme son nombre de dialectes couverts
        """
        df = self.df_wide.set_index("french_term")
        nb_dialectes = df.notna() & (df != "")
        nb_dialectes = nb_dialectes.sum(axis=1).reset_index()
        nb_dialectes.columns = ["french_term", "nb_dialectes"]
        nb_dialectes = nb_dialectes.sort_values("nb_dialectes", ascending=False)

        top20 = nb_dialectes.head(20)
        bottom20 = nb_dialectes[nb_dialectes["nb_dialectes"] > 0].tail(20)
        all_dialects = nb_dialectes[nb_dialectes["nb_dialectes"] == len(self.lang_cols)]

        print("Top 20 termes présents dans le plus de dialectes :")
        for _, row in top20.iterrows():
            print(f"  {row['french_term']:30s} → {row['nb_dialectes']} dialectes")
        print(f"\nBottom 20 termes (moins représentés) :")
        for _, row in bottom20.iterrows():
            print(f"  {row['french_term']:30s} → {row['nb_dialectes']} dialectes")
        print(f"\nTermes présents dans TOUS les dialectes : {len(all_dialects)}")
        for _, row in all_dialects.iterrows():
            print(f"  {row['french_term']}")

        return nb_dialectes

    def word_length_stats(self) -> pd.DataFrame:
        """
        B2 — Statistiques descriptives de la longueur des mots par dialecte.

        Calcule pour chaque dialecte (et le français) :
          - moyenne, médiane, min, max, écart-type de la longueur des mots

        Returns:
            DataFrame contenant les stats par dialecte
        """
        rows = []
        for code, name in self.LANG_NAMES.items():
            mots = self.df_wide[code].dropna()
            mots = mots[mots != ""].astype(str)
            lengths = mots.str.len()
            if len(lengths) > 0:
                rows.append({
                    "dialecte": name,
                    "moyenne": round(lengths.mean(), 2),
                    "mediane": lengths.median(),
                    "min": lengths.min(),
                    "max": lengths.max(),
                    "ecart_type": round(lengths.std(), 2),
                })
        # Statistiques pour le français
        fr_lengths = self.df_wide["french_term"].astype(str).str.len()
        rows.append({
            "dialecte": "Français",
            "moyenne": round(fr_lengths.mean(), 2),
            "mediane": fr_lengths.median(),
            "min": fr_lengths.min(),
            "max": fr_lengths.max(),
            "ecart_type": round(fr_lengths.std(), 2),
        })
        df_stats = pd.DataFrame(rows)
        print(df_stats.to_string(index=False))
        return df_stats

    # ------------------------------------------------------------------
    # B3. Visualisations
    # ------------------------------------------------------------------

    def plot_pairs_per_dialect(self):
        """
        B3 — Graphique 1 : Barres horizontales du nombre de paires par dialecte.

        Sauvegarde : figures/paires_par_dialecte.png
        """
        counts = self.df_long["target_lang"].value_counts().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        counts.plot(kind="barh", ax=ax, color="steelblue", edgecolor="black")
        ax.set_xlabel("Nombre de paires de traduction")
        ax.set_ylabel("Dialecte")
        ax.set_title("Nombre de paires de traduction par dialecte")
        for i, v in enumerate(counts):
            ax.text(v + 20, i, str(v), va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "paires_par_dialecte.png", dpi=150)
        plt.close()

    def plot_coverage_heatmap(self, sample_size: int = 100):
        """
        B3 — Graphique 2 : Heatmap binaire de couverture terme × dialecte.

        Affiche une matrice où chaque cellule vaut 1 (traduit) ou 0 (absent).
        Seuls les 100 premiers termes sont échantillonnés pour la lisibilité.

        Sauvegarde : figures/heatmap_couverture.png

        Args:
            sample_size : Nombre de termes à échantillonner (défaut: 100)
        """
        df = self.df_wide.set_index("french_term")
        binary = df.notna() & (df != "")
        binary = binary.astype(int)
        sample = binary.head(sample_size)
        plt.figure(figsize=(12, max(6, sample_size // 4)))
        sns.heatmap(sample, cmap="Blues", cbar_kws={"label": "Présent (1) / Absent (0)"})
        plt.title(f"Couverture des traductions (échantillon de {sample_size} termes)")
        plt.xlabel("Dialecte")
        plt.ylabel("Terme français")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "heatmap_couverture.png", dpi=150)
        plt.close()

    def plot_word_length_boxplot(self, dialects: list[str] = None):
        """
        B3 — Graphique 3 : Boxplot de la longueur des mots par dialecte.

        Compare les 5 premiers dialectes + le français.

        Sauvegarde : figures/longueur_mots_boxplot.png

        Args:
            dialects : Liste des codes dialectes à inclure (défaut: 5 premiers)
        """
        if dialects is None:
            dialects = list(self.LANG_NAMES.keys())[:5]
        data = []
        labels = []
        for code in dialects:
            mots = self.df_wide[code].dropna()
            mots = mots[mots != ""].astype(str)
            data.append(mots.str.len())
            labels.append(self.LANG_NAMES[code])
        fr_lengths = self.df_wide["french_term"].astype(str).str.len()
        data.append(fr_lengths)
        labels.append("Français")

        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = ["lightblue"] * len(dialects) + ["lightgreen"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_ylabel("Longueur (nombre de caractères)")
        ax.set_title("Distribution de la longueur des mots par dialecte")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "longueur_mots_boxplot.png", dpi=150)
        plt.close()

    def plot_wordcloud(self, dialect_code: str):
        """
        B3 — Graphique 4/5 : Nuage de mots pour un dialecte donné.

        Les mots les plus fréquents apparaissent en plus gros.

        Sauvegarde : figures/wordcloud_{dialect_code}.png

        Args:
            dialect_code : Code du dialecte (ex: "Mb" pour Mbay)
        """
        try:
            from wordcloud import WordCloud
        except ImportError:
            print("wordcloud non installé. Saute le nuage de mots.")
            return
        mots = self.df_wide[dialect_code].dropna()
        mots = mots[mots != ""].astype(str)
        text = " ".join(mots)
        if not text.strip():
            print(f"Aucun mot pour {dialect_code}")
            return
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Nuage de mots — {self.LANG_NAMES.get(dialect_code, dialect_code)}")
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"wordcloud_{dialect_code}.png", dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # Méthode principale
    # ------------------------------------------------------------------

    def run(self):
        """
        Exécute le pipeline complet d'exploration :
          B1 → B2 → B3 (5 graphiques)
        """
        print("=== B1. Statistiques de base ===")
        self.basic_stats()

        print("\n=== B2. Couverture ===")
        self.coverage_analysis()

        print("\n=== B2. Longueur des mots ===")
        self.word_length_stats()

        print("\n=== B3. Visualisations ===")
        self.plot_pairs_per_dialect()
        self.plot_coverage_heatmap()
        self.plot_word_length_boxplot()

        # Nuages de mots pour deux dialectes représentatifs
        self.plot_wordcloud("Mb")
        self.plot_wordcloud("Ngb")

        print(f"\nGraphiques sauvegardés dans : {self.figures_dir.resolve()}")


# ------------------------------------------------------------------
# Point d'entrée
# ------------------------------------------------------------------

if __name__ == "__main__":
    explorer = SaraExplorer(
        wide_csv="data/sara_wide.csv",
        long_csv="data/sara_long.csv",
        figures_dir="figures/",
    )
    explorer.run()
