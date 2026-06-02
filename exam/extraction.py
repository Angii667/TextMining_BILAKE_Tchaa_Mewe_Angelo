"""
Partie A — Extraction du lexique Sara depuis le PDF.

Ce script extrait les données du dictionnaire phonétique Sara sur la plage
ciblée des pages 11 à 77, corrige les anomalies d'encodage identifiées par
l'audit global des caractères, et génère les exports aux formats attendus.

Bibliothèques requises : pdfplumber, pandas
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
    produit trois formats de fichiers CSV nettoyés en UTF-8 Standard.
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
    # Méthode de nettoyage globale intégrant toutes les tactiques d'encodage
    # ------------------------------------------------------------------
    def clean_sara_text(self, text: str) -> str:
        """
        Nettoie et convertit l'intégralité des caractères corrompus (Legacy Font)
        détectés sur la plage cible du PDF vers l'UTF-8 international standard.
        """
        if not text:
            return text

        # 1. Remplacements contextuels et exceptions prioritaires
        text = text.replace("kîl", "kōl")
        text = text.replace("Éy", "ə́y")
        text = text.replace("nd¸g", "ndōg")
        text = text.replace("îl", "ɔ̀l")

        # 2. Table de conversion exhaustive (basée sur l'audit complet du PDF)
        font_corrections = {
            # --- CONSONNES INJECTIVES ET SPÉCIALES ---
            "ß": "ɓ",       # B injectif (Découvert lors de l'audit des 66 pages)
            "÷": "ɗ",       # D injectif
            "ð": "ɗ",       # Variante de D injectif / diacritique
            "ñ": "ɲ",       # n palatal (gn)
            "Ñ": "Ɲ",       # N palatal majuscule
            "…": "ɲ",       # Autre encodage du n palatal
            "È": "ṛ",       # r rétroflexe
            "°": "ṛ",       # r rétroflexe
            "®": "r̄",       # r avec macron (long)
            "Ž": "č",       # c caron (tch)

            # --- VOYELLES CONVERTIES DE LA POLICE HÉRITÉE ---
            "û": "mo",      # Note: souvent mappé sur o long ou macron 'ō'
            "ö": "ō",
            "î": "ō",
            "ä": "ā",
            "æ": "ā",
            "ü": "ū",
            "ï": "ī",
            "ë": "ē",
            "‡": "í",
            
            # --- VOYELLES OUVERTES ET ACCENTS SPÉCIFIQUES ---
            "Æ": "ɛ̀",      # Epsilon ouvert + accent grave
            "£": "Ɛ",      # Epsilon ouvert majuscule
            "Õ": "ɔ̀",      # O ouvert + accent grave
            "ÿ": "ȳ",      # y avec macron
            "ý": "ý",      # y avec accent aigu
            "Û": "w̄",      # w avec macron
            "Ç": "ó",      # o accent aigu
            "ô": "ó",      # o accent aigu
            "É": "é",      # e accent aigu (hors contexte Éy)

            # --- VOYELLES NASALISÉES (TILDE / CROCHET SOUSCRIT) ---
            "¡": "ı̰",      # i sans point avec tilde souscrit
            "Ð": "ɛ̰̀",     # epsilon ouvert grave avec tilde souscrit
            "¼": "à̰",      # a grave avec tilde souscrit
            "Ÿ": "ḛ̀",      # e grave avec tilde souscrit
            "Ï": "ḛ́",      # e aigu avec tilde souscrit
            
            # --- NETTOYAGE DES SCORIES DE RENDU ---
            "¸": "",       # Suppression du résidu de cédille déplacée
            " ¸": "",
        }
        
        for legacy_char, unicode_char in font_corrections.items():
            text = text.replace(legacy_char, unicode_char)
            
        # 3. Nettoyage final des espaces et ponctuations parasites de fin de ligne
        text = text.rstrip(" :")
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ------------------------------------------------------------------
    # Méthode 1 : lecture du PDF (Ciblée strictement sur les pages 11 à 77)
    # ------------------------------------------------------------------
    def extract(self) -> list[dict]:
        """
        Lit le PDF uniquement sur la plage des pages 11 à 77 (indices 10 à 77)
        et extrait toutes les paires (terme_français, code_dialecte, mot_sara).
        """
        lignes_brutes = []
        entree_courante = None

        # Pattern pour détecter les lignes de traduction (ex: Mb=...)
        lang_regex = r'\b(' + '|'.join(SARA_LANGS.keys()) + r')\s*=\s*'

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Fichier PDF non trouvé : {self.pdf_path}")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            # Sécurisation du slice : de la page 11 (index 10) à la page 77 (index 76 inclus)
            pages_du_dictionnaire = pdf.pages[10:77]
            
            print(f"Extraction ciblée en cours sur les pages 11 à 77 du document.")

            for page in pages_du_dictionnaire:
                width = page.width
                height = page.height

                # Séparation géométrique en deux colonnes indépendantes
                domaine_gauche = (0, 0, width/2, height)
                domaine_droite = (width/2, 0, width, height)

                col_gauche = page.within_bbox(domaine_gauche).extract_text()
                col_droite = page.within_bbox(domaine_droite).extract_text()

                for col_texte in [col_gauche, col_droite]:
                    if not col_texte:
                        continue

                    lignes = col_texte.split("\n")
                    for ligne in lignes:
                        ligne = ligne.strip()
                        if not ligne:
                            continue

                        # Ignorer les métadonnées de page
                        if ligne.isdigit() or ligne.lower().startswith("sara languages lexicon"):
                            continue

                        # Détection des tags de dialectes dans la ligne
                        contient_lang = any(f'{lang}=' in ligne for lang in SARA_LANGS)

                        if contient_lang:
                            if entree_courante is not None:
                                correspondances = list(re.finditer(lang_regex, ligne))
                                for i, correspondance in enumerate(correspondances):
                                    lang_code = correspondance.group(1)
                                    start_idx = correspondance.end()
                                    end_idx = correspondances[i+1].start() if i+1 < len(correspondances) else len(ligne)

                                    mot = ligne[start_idx:end_idx].strip()
                                    if mot:
                                        # Application de la correction d'encodage
                                        mot = self.clean_sara_text(mot)
                                        if mot:
                                            entree_courante[lang_code] = mot
                        else:
                            # Nouvelle entrée (Terme pivot en Français)
                            if entree_courante is not None and len(entree_courante) > 1:
                                lignes_brutes.append(entree_courante)
                            entree_courante = {"french_term": ligne}
                    
            # Enregistrement de la dernière entrée du dictionnaire
            if entree_courante is not None and len(entree_courante) > 1:
                lignes_brutes.append(entree_courante)
        
        return lignes_brutes

    # ------------------------------------------------------------------
    # Méthode 2 : Transformation au format LARGE
    # ------------------------------------------------------------------
    def to_wide(self) -> pd.DataFrame:
        """
        Transforme les données extraites en format large (un terme par ligne).
        """
        if not self.lignes_brutes:
            colonnes_ordonnees = ["french_term"] + list(SARA_LANGS.keys())
            return pd.DataFrame(columns=colonnes_ordonnees)
        
        df_wide = pd.DataFrame(self.lignes_brutes)

        # S'assurer que toutes les colonnes de dialectes existent
        for langue in SARA_LANGS.keys():
            if langue not in df_wide.columns:
                df_wide[langue] = None

        colonnes_ordonnees = ["french_term"] + list(SARA_LANGS.keys())
        df_wide = df_wide[colonnes_ordonnees]

        chemin_sortie = self.output_dir / "sara_wide.csv"
        df_wide.to_csv(chemin_sortie, index=False, encoding="utf-8-sig")
        
        return df_wide

    # ------------------------------------------------------------------
    # Méthode 3 : Transformation au format LONG
    # ------------------------------------------------------------------
    def to_long(self) -> pd.DataFrame:
        """
        Transforme les données en format long (paires de traduction empilées).
        """
        format_long_data = []

        for ligne in self.lignes_brutes:
            terme_francais = ligne.get("french_term")
            if not terme_francais:
                continue

            for langue_codee, nom_langue in SARA_LANGS.items():
                mot = ligne.get(langue_codee)
                if pd.notna(mot) and str(mot).strip() != "":
                    format_long_data.append({
                        "source_lang": "French",
                        "source_word": terme_francais,
                        "target_lang": nom_langue,
                        "target_word": str(mot).strip()
                    })
        
        df_long = pd.DataFrame(format_long_data, columns=["source_lang", "source_word", "target_lang", "target_word"])
        
        chemin_sortie = self.output_dir / "sara_long.csv"
        df_long.to_csv(chemin_sortie, index=False, encoding="utf-8-sig")

        return df_long

    # ------------------------------------------------------------------
    # Méthode 4 : Génération des fichiers individuels par dialecte
    # ------------------------------------------------------------------
    def to_dialect_pairs(self) -> dict[str, pd.DataFrame]:
        """
        Produit un fichier CSV propre à deux colonnes pour chaque dialecte.
        """
        pairs_dir = self.output_dir / "pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)

        dialecte_dfs = {}

        for langue_codee, nom_langue in SARA_LANGS.items():
            pairs_data = []

            for ligne in self.lignes_brutes:
                terme_francais = ligne.get("french_term")
                mot = ligne.get(langue_codee)

                if terme_francais and pd.notna(mot) and str(mot).strip() != "":
                    pairs_data.append({
                        nom_langue.lower(): str(mot).strip(),
                        "français": terme_francais
                    })
            
            df_dialect = pd.DataFrame(pairs_data, columns=[nom_langue.lower(), "français"])

            nom_fichier = f"{nom_langue.lower()}_french.csv"
            chemin_sortie = pairs_dir / nom_fichier
            df_dialect.to_csv(chemin_sortie, index=False, encoding="utf-8-sig")

            dialecte_dfs[nom_langue] = df_dialect
        
        return dialecte_dfs

    # ------------------------------------------------------------------
    # Méthode principale d'orchestration
    # ------------------------------------------------------------------
    def run(self):
        """
        Exécute l'intégralité du pipeline
        """
        print("=== DEBUT DE L'EXTRACTION DU LEXIQUE SARA ===")
        self.lignes_brutes = self.extract()
        print(f"  [Succès] {len(self.lignes_brutes)} entrées du dictionnaire chargées.")

        df_wide = self.to_wide()
        print(f"  [Export] Format large généré : {df_wide.shape[0]} termes.")

        df_long = self.to_long()
        print(f"  [Export] Format long généré : {len(df_long)} paires associées.")

        dialect_dfs = self.to_dialect_pairs()
        print(f"  [Export] {len(dialect_dfs)} fichiers par dialecte créés dans 'data/pairs/'.")
        print("=== PIPELINE TERMINE AVEC SUCCÈS ===")


# ------------------------------------------------------------------
# Point d'entrée d'exécution du script
# ------------------------------------------------------------------
if __name__ == "__main__":
    extractor = SaraExtractor(
        pdf_path="/home/broly/TextMining_BLK/data/raw/SaraLanguagesLexicon.pdf",
        output_dir="/home/broly/TextMining_BLK/data/",
    )
    extractor.run()
