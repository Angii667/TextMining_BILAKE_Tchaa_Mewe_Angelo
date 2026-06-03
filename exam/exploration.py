"""
Partie B — Exploration statistique du lexique Sara.

Complétez chaque méthode de la classe SaraExplorer.
Chaque méthode doit afficher ses résultats (print) ET
sauvegarder les graphiques dans le dossier figures/.

Bibliothèques : pandas, matplotlib, seaborn, wordcloud
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class SaraExplorer:
    """
    Analyse statistique et visualisation du lexique multilingue Sara.
    Utilise le format large (sara_wide.csv) comme source principale.
    """

    # Noms complets des dialectes (pour les graphiques)
    LANG_NAMES = {
        "Beb": "Bebote",   "Bd": "Bediondo", "Db": "Daba",
        "Gor": "Gor",      "Gu": "Gulay",    "KbN": "Kaba_Na",
        "Kbb": "Kaba",     "Lk": "Laka",     "Mb": "Mbay",
        "Mo":  "Mango",    "Nar": "Nar",      "Ngb": "Ngambay",
        "Sr":  "Sar",      "NgT": "Ngam",
    }

    def __init__(self, wide_csv: str, long_csv: str, figures_dir: str = "figures/"):
        """
        Args:
            wide_csv    : Chemin vers sara_wide.csv
            long_csv    : Chemin vers sara_long.csv
            figures_dir : Dossier de sauvegarde des graphiques
        """
        self.df_wide = pd.read_csv(wide_csv, encoding="utf-8-sig")
        self.df_long = pd.read_csv(long_csv, encoding="utf-8-sig")
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Colonnes correspondant aux dialectes (tout sauf french_term)
        self.lang_cols = [c for c in self.df_wide.columns if c != "french_term"]

    # ------------------------------------------------------------------
    # B1. Statistiques de base
    # ------------------------------------------------------------------

    def basic_stats(self) -> dict:
        """
        Calcule et affiche les statistiques fondamentales :
        - Nombre de paires par dialecte
        - Taux de couverture par dialecte (% de termes traduits)

        Returns:
            Dictionnaire avec les métriques calculées.
        """
        # Nombre de termes français uniques
        nbre_fr_uniq = self.df_wide["french_term"].nunique()
        # Nombre total de paires de traduction
        pairs_par_traduction = len(self.df_long)
        # Nombre de paires par dialectes
        nbre_pairs_par_dialecte =  self.df_wide[self.lang_cols].notna().sum()
        raise NotImplementedError("À implémenter")

    # ------------------------------------------------------------------
    # B2. Analyse du vocabulaire
    # ------------------------------------------------------------------

    def coverage_analysis(self) -> pd.DataFrame:
        """
        Pour chaque terme français, calcule le nombre de dialectes
        dans lesquels il est traduit.

        Affiche :
        - Les 20 termes présents dans le plus grand nombre de dialectes
        - Les 20 termes présents dans le plus petit nombre de dialectes
        - Les termes présents dans TOUS les dialectes (s'ils existent)

        Returns:
            DataFrame avec les colonnes : french_term, nb_dialectes
        """
        raise NotImplementedError("À implémenter")

    def word_length_stats(self) -> pd.DataFrame:
        """
        Pour chaque dialecte, calcule des statistiques sur la longueur
        des mots (en nombre de caractères) :
        - Moyenne, médiane, minimum, maximum, écart-type

        Comparez également avec la longueur des termes français.

        Returns:
            DataFrame avec une ligne par dialecte et les statistiques.
        """
        raise NotImplementedError("À implémenter")

    # ------------------------------------------------------------------
    # B3. Visualisations
    # ------------------------------------------------------------------

    def plot_pairs_per_dialect(self):
        """
        Crée un diagramme en barres horizontales du nombre de paires
        de traduction par dialecte, trié du plus au moins représenté.

        Sauvegarde : figures/paires_par_dialecte.png
        """
        raise NotImplementedError("À implémenter")

    def plot_coverage_heatmap(self, sample_size: int = 100):
        """
        Crée une carte de chaleur (heatmap) montrant la présence (1)
        ou l'absence (0) de traduction pour un échantillon de termes
        français (lignes) × dialectes (colonnes).

        Conseil : limitez à sample_size termes pour la lisibilité.

        Sauvegarde : figures/heatmap_couverture.png
        """
        raise NotImplementedError("À implémenter")

    def plot_word_length_boxplot(self, dialects: list[str] = None):
        """
        Crée un boxplot comparant la distribution de la longueur des
        mots pour plusieurs dialectes et pour le français.

        Args:
            dialects : Liste de codes de dialectes à afficher.
                       Par défaut : les 5 premiers.

        Sauvegarde : figures/longueur_mots_boxplot.png
        """
        raise NotImplementedError("À implémenter")

    def plot_wordcloud(self, dialect_code: str):
        """
        Crée un nuage de mots pour le dialecte spécifié.

        Args:
            dialect_code : Code du dialecte (ex. "Mb" pour Mbay)

        Sauvegarde : figures/wordcloud_{dialect_code}.png
        """
        raise NotImplementedError("À implémenter")

    # ------------------------------------------------------------------
    # Méthode principale
    # ------------------------------------------------------------------

    def run(self):
        """
        Exécute l'exploration complète dans l'ordre :
        1. Statistiques de base
        2. Analyse de couverture
        3. Statistiques de longueur
        4. Graphiques
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

        # Générez des nuages de mots pour au moins 2 dialectes
        self.plot_wordcloud("Mb")   # Mbay
        self.plot_wordcloud("Ngb")  # Ngambay

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
