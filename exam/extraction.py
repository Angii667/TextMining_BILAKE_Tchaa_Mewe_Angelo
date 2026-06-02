"""
Partie A — Extraction du lexique Sara depuis le PDF.

Complétez chaque méthode de la classe SaraExtractor.
Ne modifiez pas les signatures des méthodes ni les noms des fichiers de sortie.

Bibliothèques suggérées : pdfplumber, pdfminer.six, pandas, re, pathlib
"""

import re
from pathlib import Path
import pdfplumber
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
        self.lignes_brutes: list[dict] = []

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
        lignes_brutes = []
        entree_courante = None

        #Pattern pour détecter les lignes de traduction
        lang_regex = r'\b(' + '|'.join(SARA_LANGS.keys()) + r')\s*=\s*'

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Fichier PDF non trouvé : {self.pdf_path}")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                width = page.width
                height = page.height

                #Definition des des limites de collision pour separer les deux colonnes
                domaine_gauche = (0, 0, width/2, height)
                domaine_droite = (width/2, 0, width, height)

                #Définition du texte par zone géographique de page
                col_gauche = page.within_bbox(domaine_gauche).extract_text()
                col_droite = page.within_bbox(domaine_droite).extract_text()

                for col_texte in [col_gauche, col_droite]:
                    if not col_texte:
                        continue

                    lignes = col_texte.split("\n") # saut de ligne comme séparateur
                    for ligne in lignes:
                        ligne = ligne.strip()
                        if not ligne:
                            continue

                        #Nettoyage, on va ignorer les numeros de pages isolés ou les en-tetes évidents
                        if ligne.isdigit() or ligne.lower().startswith("sara languages lexicon"):
                            continue

                        #Détection de la présence d'au moins un dialect codé
                        contient_lang = any(f'{lang}=' in ligne for lang in SARA_LANGS)

                        if contient_lang:
                            if entree_courante is not None:
                                #trouver toutes les positions des tags "Dialecte=" sur la ligne
                                correspondances = list(re.finditer(lang_regex, ligne))
                                for i, correspondance in enumerate(correspondances):
                                    lang_code = correspondance.group(1)
                                    start_idx = correspondance.end()
                                    #Le mot s'arrete au prochain match ou à la fin de la ligne
                                    end_idx = correspondances[i+1].start() if i+1<len(correspondances) else len(ligne)

                                    mot = ligne[start_idx:end_idx].strip()
                                    if mot:
                                        entree_courante[lang_code] = mot #On obtient au final entree_courante un dictionnaire qui a chaque dialecte à ses mots correspondant
                        
                        else:
                            #C'est une ligne de terme francais (nouvelle entrée)
                            if entree_courante is not None and len(entree_courante)>1:
                                lignes_brutes.append(entree_courante)
                            entree_courante = {"french_term":ligne}
                    
            #On n'oublie pas de sauvegarder le tout dernier élément traité
            if entree_courante is not None and len(entree_courante) > 1:
                lignes_brutes.append(entree_courante)
        
        return lignes_brutes

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
        if not self.lignes_brutes:
            colonnes_ordonnees = ["french_term"] + list(SARA_LANGS.keys())
            return pd.DataFrame(columns= colonnes_ordonnees)
        
        df_wide = pd.DataFrame(self.lignes_brutes)

        #Garantir de toutes les colonnes de dialectes meme vides existent
        for langue in SARA_LANGS.keys():
            if langue not in df_wide.columns:
                df_wide[langue] = None

        #On va ordonner rigoureusement les colonnes selon la consigne
        colonnes_ordonnees = ["french_term"] + list(SARA_LANGS.keys())
        df_wide = df_wide[colonnes_ordonnees]

        #Sauvegarde
        chemin_sortie = self.output_dir / "sara_wide.csv"
        df_wide.to_csv(chemin_sortie, index=False, encoding="utf-8-sig")
        
        return df_wide

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

        Que les paires où le mot Sara n'est pas vide sont incluses

        Sauvegarde : output_dir/sara_long.csv (encodage utf-8-sig)

        Returns:
            DataFrame au format long.
        """
        format_long_data = []

        for ligne in self.lignes_brutes:
            terme_francais = ligne.get("french_term")
            if not terme_francais:
                continue

            for langue_codee, nom_langue in SARA_LANGS.items():
                mot = ligne.get(langue_codee)
                #On ne conserve que si le mot existe et n'est pas vide
                if pd.notna(mot) and str(mot).strip() != "":
                    format_long_data.append({
                        "source_lang":"French",
                        "source_word": terme_francais,
                        "target_lang": nom_langue,
                        "target_word": str(mot).strip()
                    })
        
        df_long = pd.DataFrame(format_long_data, columns = ["source_lang","source_word", "target_lang", "target_word"])
        #Sauvegarde
        chemin_sortie = self.output_dir / "sara_long.csv"
        df_long.to_csv(chemin_sortie, index=False, encoding="utf-8-sig")

        return df_long
    # ------------------------------------------------------------------
    # Méthode 4 : fichiers par dialecte
    # ------------------------------------------------------------------

    def to_dialect_pairs(self) -> dict[str, pd.DataFrame]:
        """
        Produit un fichier CSV à deux colonnes pour chaque dialecte.

        Pour le dialecte "Mbay" (code "Mb"), le fichier aura :
        - Nom : pairs/mbay_french.csv
        - Colonnes : mbay | français

        Seules les lignes où le mot Sara n'est pas vide seront conservées.

        Sauvegarde : output_dir/pairs/{nom_dialecte_lower}_french.csv

        Returns:
            Dictionnaire {nom_dialecte: DataFrame}
        """
        pairs_dir = self.output_dir/"pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)

        dialecte_dfs = {}

        for langue_codee, nom_langue in SARA_LANGS.items():
            pairs_data = []

            for ligne in self.lignes_brutes:
                terme_francais = ligne.get("french_term")
                mot = ligne.get(langue_codee)

                if terme_francais and pd.notna(mot) and str(mot).strip() !="":
                    pairs_data.append({
                        nom_langue.lower(): str(mot).strip(),
                        "français": terme_francais
                    })
            #Géneration du DataFrame pour le dialecte en cours
            df_dialect = pd.DataFrame(pairs_data, columns=[nom_langue.lower(), "français"])

            #Sauvegarde
            nom_fichier = f"{nom_langue.lower()}_french.csv"
            chemin_sortie = pairs_dir / nom_fichier
            df_dialect.to_csv(chemin_sortie, index=False, encoding="utf-8-sig")

            dialecte_dfs[nom_langue] = df_dialect
        
        return dialecte_dfs

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
        self.lignes_brutes = self.extract()
        print(f"  {len(self.lignes_brutes)} entrées extraites")

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
        pdf_path="/home/broly/TextMining_BILAKE_Tchaa_Mèwè_Angelo/data/raw/SaraLanguagesLexicon.pdf",
        output_dir="/home/broly/TextMining_BILAKE_Tchaa_Mèwè_Angelo/data/",
    )
    extractor.run()
