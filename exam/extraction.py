"""
Partie A — Extraction du lexique Sara depuis le PDF.

Complétez chaque méthode de la classe SaraExtractor.
Ne modifiez pas les signatures des méthodes ni les noms des fichiers de sortie.

Bibliothèques suggérées : pdfplumber, pdfminer.six, pandas, re, pathlib
"""

import re
from pathlib import Path

import pandas as pd


# Codes des 14 dialectes Sara et leurs noms complets
SARA_LANGS = {
    "Beb": "Bebote",
    "Bd":  "Bediondo",
    "Db":  "Daba",
    "Gor": "Gor",
    "Gu":  "Gulay",
    "KbN": "Kaba_Na",
    "Kbb": "Kaba",
    "Lk":  "Laka",
    "Mb":  "Mbay",
    "Mo":  "Mango",
    "Nar": "Nar",
    "Ngb": "Ngambay",
    "Sr":  "Sar",
    "NgT": "Ngam",
}


class SaraExtractor:
    """
    Extrait les données du fichier SaraLanguagesLexicon.pdf et
    produit trois formats de fichiers CSV.
    """

    def __init__(self, pdf_path: str, output_dir: str = "data/"):
        """
        Initialise l'extracteur.

        Args:
            pdf_path   : Chemin vers le fichier SaraLanguagesLexicon.pdf
            output_dir : Dossier de sortie pour les fichiers CSV
        """
        self.pdf_path   = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sera rempli après l'appel à extract()
        self.raw_rows: list[dict] = []

    # ------------------------------------------------------------------
    # Méthode 1 : lecture du PDF
    # ------------------------------------------------------------------

    def extract(self) -> list[dict]:
        """
        Lit le PDF page par page et extrait toutes les paires
        (terme_français, code_dialecte, mot_sara).

        Conseils :
        - Chaque page est divisée en deux colonnes. Pensez à séparer
          la partie gauche et la partie droite avant de parser.
        - Une entrée se compose d'une ligne de terme français suivie
          d'une ou plusieurs lignes du type : Mb=mot Gor=mot Sr=mot …
        - Ignorez les numéros de page, titres et lignes vides.

        Returns:
            Liste de dicts, chacun ayant la structure :
            {"french_term": str, "Mb": str, "Gor": str, ...}
            (seuls les dialectes présents sont inclus pour chaque terme)
        """
        raise NotImplementedError("À implémenter")

    # ------------------------------------------------------------------
    # Méthode 2 : format large
    # ------------------------------------------------------------------

    def to_wide(self) -> pd.DataFrame:
        """
        Transforme les données extraites en format large :
        une ligne par terme français, une colonne par dialecte.

        Les cellules vides (dialecte non traduit) sont représentées
        par une chaîne vide ou NaN.

        Ordre des colonnes : ["french_term", "Beb", "Bd", ..., "NgT"]

        Sauvegarde : output_dir/sara_wide.csv (encodage utf-8-sig)

        Returns:
            DataFrame au format large.
        """
        raise NotImplementedError("À implémenter")

    # ------------------------------------------------------------------
    # Méthode 3 : format long
    # ------------------------------------------------------------------

    def to_long(self) -> pd.DataFrame:
        """
        Transforme les données en format long :
        une ligne par paire (terme_français, dialecte, mot_sara).

        Colonnes : source_lang, source_word, target_lang, target_word

        - source_lang  : toujours "French"
        - source_word  : le terme français
        - target_lang  : le nom complet du dialecte (ex. "Mbay")
        - target_word  : le mot dans ce dialecte

        N'incluez que les paires où le mot Sara n'est pas vide.

        Sauvegarde : output_dir/sara_long.csv (encodage utf-8-sig)

        Returns:
            DataFrame au format long.
        """
        raise NotImplementedError("À implémenter")

    # ------------------------------------------------------------------
    # Méthode 4 : fichiers par dialecte
    # ------------------------------------------------------------------

    def to_dialect_pairs(self) -> dict[str, pd.DataFrame]:
        """
        Produit un fichier CSV à deux colonnes pour chaque dialecte.

        Pour le dialecte "Mbay" (code "Mb"), le fichier aura :
        - Nom : pairs/mbay_french.csv
        - Colonnes : mbay | français

        Seules les lignes où le mot Sara n'est pas vide sont conservées.

        Sauvegarde : output_dir/pairs/{nom_dialecte_lower}_french.csv

        Returns:
            Dictionnaire {nom_dialecte: DataFrame}
        """
        raise NotImplementedError("À implémenter")

    # ------------------------------------------------------------------
    # Méthode principale
    # ------------------------------------------------------------------

    def run(self):
        """
        Exécute le pipeline complet :
        1. Extraction depuis le PDF
        2. Sauvegarde format large
        3. Sauvegarde format long
        4. Sauvegarde fichiers par dialecte
        """
        print("Extraction en cours...")
        self.raw_rows = self.extract()
        print(f"  {len(self.raw_rows)} entrées extraites")

        df_wide = self.to_wide()
        print(f"  Format large : {df_wide.shape[0]} termes × {df_wide.shape[1]} colonnes")

        df_long = self.to_long()
        print(f"  Format long  : {len(df_long)} paires de traduction")

        dialect_dfs = self.to_dialect_pairs()
        print(f"  {len(dialect_dfs)} fichiers par dialecte générés")

        print("Terminé.")


# ------------------------------------------------------------------
# Point d'entrée
# ------------------------------------------------------------------

if __name__ == "__main__":
    extractor = SaraExtractor(
        pdf_path="SaraLanguagesLexicon.pdf",
        output_dir="data/",
    )
    extractor.run()
