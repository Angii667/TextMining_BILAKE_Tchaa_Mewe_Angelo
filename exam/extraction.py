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
    # Méthode de nettoyage globale validée par l'audit des 66 pages
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

        # 2. Table de conversion exhaustive (mapping SIL Legacy vers Unicode)
        font_corrections = {
            # --- VOYELLES OUVERTES ET CENTRALES SARA ---
            "Æ": "ɛ",       # E ouvert (ex: kÆm -> kɛm)
            "æ": "ɛ",       # E ouvert ou Schwa selon la variante dialectale
            "Õ": "ɔ",       # O ouvert (ex: kÕ-ndû-g¸ -> kɔ̄-ndū-gə)
            "õ": "ɔ",       # O ouvert (variante minuscule)
            "‡": "ɨ",       # Voyelle centrale haute barrée (ex: k‡rª -> kɨ̄rā)
            "¸": "ə",       # Schwa / Voyelle centrale moyenne (ex: màd¸ -> màdə)

            # --- CONSONNES SPÉCIALES ET LIQUIDES ---
            "£": "l",       # L standard ou liquide flappée (ex: yÆ£ -> yɛl)
            "¥": "l",       # L ou R flappé selon le dialecte
            "®": "r",       # R rétroflexe / battu (ex: gÆ® -> gɛr)
            "÷": "ɽ",       # R battu / flappé spécifique (ex: ÷á -> ɽá)
            "•": "r",       # Scorie d'accent ou R flappé (ex: bö• -> bōr)

            # --- CORRECTIONS DES PREMIÈRES PAGES ---
            "5": "ɔ",       # Chiffre utilisé pour le O ouvert
            "6": "ɓ",       # Chiffre utilisé pour l'implosive bilabiale
            "3": "ɓ",       # Chiffre parfois substitué à l'implosive majuscule
            "王": "ī",      # Idéogramme parasite, corruption de 'i' à ton moyen
            "±": "ī",       # Signe plus/moins substitué pour un 'i' centralisé

            # --- ACCENTS ET DIACRITIQUES DE TONS COMBINÉS ---
            "ä": "ā",       # 'a' avec ton moyen (macron)
            "ë": "ē",       # 'e' avec ton moyen
            "ï": "ī",       # 'i' avec ton moyen ou haut
            "ö": "ó",       # 'o' avec ton haut (ex: òö -> òó)
            "ü": "ū",       # 'u' avec ton moyen ou haut
            "û": "ú",       # 'u' avec ton haut (ex: ndû -> ndú)
            "î": "í",       # 'i' avec ton haut ou descendant
            "ô": "ó",       # 'o' avec ton haut
            "ª": "á",       # Exposant 'a' traduisant un ton haut
            "º": "ɔ́",       # Degré traduisant un O ouvert avec ton haut
            "Ç": "ɔ́",       # C cédille traduisant un O ouvert majuscule ou accentué

            # --- NETTOYAGE DES SCORIES DE PARSING ---
            " :": "",       # Suppression des deux-points parasites en fin de mot
        }

        for legacy_char, unicode_char in font_corrections.items():
            text = text.replace(legacy_char, unicode_char)

        # 3. Nettoyage final des espaces et ponctuations parasites
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ------------------------------------------------------------------
    # Méthode 1 : lecture du PDF (Ciblée sur les pages 11 à 77)
    # ------------------------------------------------------------------
    def extract(self) -> list[dict]:
        """
        Lit le PDF uniquement sur la plage des pages 11 à 77 (indices 10 à 77)
        et extrait toutes les paires (terme_français, code_dialecte, mot_sara).
        """
        lignes_brutes = []
        entree_courante = None

        # Pattern pour détecter les lignes de traduction
        lang_regex = r'\b(' + '|'.join(SARA_LANGS.keys()) + r')\s*=\s*'

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Fichier PDF non trouvé : {self.pdf_path}")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            # Sécurisation du slice : de la page 11 (index 10) à la page 77 (index 76 inclus)
            pages_du_dictionnaire = pdf.pages[10:77]
            
            print(f"  [Analyse] Extraction ciblée sur {len(pages_du_dictionnaire)} pages (Pages 11 à 77).")

            for page in pages_du_dictionnaire:
                width = page.width
                height = page.height

                # Séparation en deux colonnes strictes
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

                        # Ignorer les en-têtes et numéros de pages
                        if ligne.isdigit() or ligne.lower().startswith("sara languages lexicon"):
                            continue

                        # Détection des tags dialectes
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
                                        mot = self.clean_sara_text(mot)
                                        if mot:
                                            entree_courante[lang_code] = mot
                        else:
                            # Nouvelle entrée (Terme en Français)
                            if entree_courante is not None and len(entree_courante) > 1:
                                lignes_brutes.append(entree_courante)
                            entree_courante = {"french_term": ligne}
                    
            if entree_courante is not None and len(entree_courante) > 1:
                lignes_brutes.append(entree_courante)
        
        return lignes_brutes

    # ------------------------------------------------------------------
    # Méthode 2 : format large
    # ------------------------------------------------------------------
    def to_wide(self) -> pd.DataFrame:
        """
        Transforme les données extraites en format large (un terme par ligne).
        """
        if not self.lignes_brutes:
            colonnes_ordonnees = ["french_term"] + list(SARA_LANGS.keys())
            return pd.DataFrame(columns=colonnes_ordonnees)
        
        df_wide = pd.DataFrame(self.lignes_brutes)

        for langue in SARA_LANGS.keys():
            if langue not in df_wide.columns:
                df_wide[langue] = None

        colonnes_ordonnees = ["french_term"] + list(SARA_LANGS.keys())
        df_wide = df_wide[colonnes_ordonnees]

        chemin_sortie = self.output_dir / "sara_wide.csv"
        df_wide.to_csv(chemin_sortie, index=False, encoding="utf-8-sig")
        
        return df_wide

    # ------------------------------------------------------------------
    # Méthode 3 : format long
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
    # Méthode 4 : fichiers individuels par dialecte
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
    # Méthode principale de contrôle
    # ------------------------------------------------------------------
    def run(self):
        """
        Exécute le pipeline complet.
        """
        self.lignes_brutes = self.extract()
        print(f"  {len(self.lignes_brutes)} entrées de dictionnaire extraites avec succès.")

        df_wide = self.to_wide()
        print(f"  Format large : {df_wide.shape[0]} termes × {df_wide.shape[1]} colonnes")

        df_long = self.to_long()
        print(f"  Format long  : {len(df_long)} paires de traduction générées.")

        dialect_dfs = self.to_dialect_pairs()
        print(f"  {len(dialect_dfs)} fichiers individuels par dialecte exportés.")



# ------------------------------------------------------------------
# Point d'entrée
# ------------------------------------------------------------------
if __name__ == "__main__":
    extractor = SaraExtractor(
        pdf_path="/home/broly/TextMining_BLK/data/raw/SaraLanguagesLexicon.pdf",
        output_dir="/home/broly/TextMining_BLK/data/",
    )
    extractor.run()