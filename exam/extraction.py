"""
Partie A — Extraction du lexique Sara depuis le PDF.

ARCHITECTURE : Pipeline OCR (pdf2image + Tesseract)
─────────────────────────────────────────────────────
Problème racine : le PDF utilise des polices héritées (Legacy Fonts) dont la
table ToUnicode interne est absente ou corrompue. Toute bibliothèque qui lit
la couche texte native du PDF (pdfplumber, pdfminer, PyMuPDF…) renverra
inévitablement du "garbage text" pour les caractères de l'Alphabet National
Tchadien (ɓ, ɗ, ɛ, ɔ, ɨ, ɲ, etc.).

SOLUTION : Ignorer complètement la couche texte native. Convertir chaque page
du PDF en image haute résolution (300 DPI) avec pdf2image, puis appliquer
Tesseract OCR (moteur LSTM, langue française) qui reconnaît les glyphes
visuellement — indépendamment de l'encodage interne du fichier.

Ce workflow est identique à la lecture d'un document scanné et produit un
texte UTF-8 propre et fidèle à l'impression originale.

Bibliothèques requises : pdf2image, pytesseract, Pillow, pandas
Dépendance système    : tesseract-ocr, tesseract-ocr-fra, poppler-utils
"""

import re
from pathlib import Path

from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Codes des 14 dialectes Sara et leurs noms complets
# ──────────────────────────────────────────────────────────────────────────────
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
    Extrait les données du fichier SaraLanguagesLexicon.pdf en utilisant un
    pipeline OCR (pdf2image → Tesseract) et produit trois formats CSV en UTF-8.

    Pourquoi l'OCR et non pdfplumber/pdfminer ?
    ─────────────────────────────────────────────
    Ces bibliothèques lisent la couche texte native du PDF. Pour des polices
    héritées sans table ToUnicode correcte, elles produisent des caractères
    corrompus (ex: "ß" au lieu de "ɓ", "÷" au lieu de "ɗ"). Corriger ces
    corruptions manuellement est une course sans fin.

    Tesseract, lui, analyse les formes visuelles des glyphes. Il ne se soucie
    pas de l'encodage interne du PDF et produit directement de l'UTF-8 propre.
    """

    # ── Configuration Tesseract ───────────────────────────────────────────────
    # --oem 3  : Utilise le moteur LSTM (le plus précis pour les diacritiques)
    # --psm 4  : Une colonne de texte de taille variable (adapté aux colonnes
    #            de dictionnaire avec entrées de longueurs différentes)
    TESSERACT_CONFIG = r"--oem 3 --psm 4"

    # Résolution DPI pour la conversion PDF → Image.
    # 300 DPI est le minimum recommandé pour distinguer les diacritiques
    # proches (ex: ɓ vs b, ɗ vs d, ɔ vs o). Augmenter à 400 si des
    # confusions persistent sur votre document spécifique.
    DPI = 300

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

    # ─────────────────────────────────────────────────────────────────────────
    # Méthode de nettoyage post-OCR (légère, ciblée sur les seules erreurs
    # résiduelles de Tesseract — pas de remapping massif de polices)
    # ─────────────────────────────────────────────────────────────────────────
    def clean_ocr_text(self, text: str) -> str:
        """
        Corrige les quelques confusions résiduelles typiques de Tesseract sur
        les caractères de l'Alphabet National Tchadien.

        Contrairement à l'ancienne approche, cette table est COURTE parce que
        Tesseract reconnaît déjà visuellement les bons caractères Unicode.
        On ne corrige ici que les erreurs de substitution optique connues
        (glyphes très proches que même un humain confondrait à basse résolution).
        """
        if not text:
            return text

        # Confusions optiques résiduelles de Tesseract sur l'ANT
        # (à enrichir au fur et à mesure des observations sur votre PDF)
        ocr_corrections = {
            # Tesseract confond parfois ces paires très proches visuellement
            "b'": "ɓ",    # b + apostrophe résiduel → b injectif
            "d'": "ɗ",    # d + apostrophe résiduel → d injectif
            "e'": "ɛ",    # e + apostrophe résiduel → epsilon ouvert
            "o'": "ɔ",    # o + apostrophe résiduel → o ouvert

            # Guillemets typographiques parfois produits à la place de l'apostrophe
            "b\u2019": "ɓ",
            "d\u2019": "ɗ",

            # Tirets parasites en fin de ligne (césures OCR)
            "-\n": "",
        }

        for mauvais, bon in ocr_corrections.items():
            text = text.replace(mauvais, bon)

        # Nettoyage des espaces multiples et ponctuations parasites de fin de ligne
        text = text.rstrip(" :")
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ─────────────────────────────────────────────────────────────────────────
    # Méthode 1 : Conversion PDF → Images puis OCR (pages 11 à 77)
    # ─────────────────────────────────────────────────────────────────────────
    def extract(self) -> list[dict]:
        """
        Convertit les pages 11 à 77 du PDF en images haute résolution, applique
        Tesseract OCR sur chaque demi-colonne, puis parse les paires
        (terme_français, code_dialecte, mot_sara).

        Le découpage géométrique en deux colonnes est conservé de l'ancienne
        approche, mais s'applique maintenant sur les images PIL avant l'OCR
        (au lieu de s'appliquer sur les zones textuelles de pdfplumber).
        """
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Fichier PDF non trouvé : {self.pdf_path}")

        # Pattern pour détecter les tags de dialectes (ex: "Mb=", "KbN=")
        lang_regex = r"\b(" + "|".join(re.escape(k) for k in SARA_LANGS.keys()) + r")\s*=\s*"

        lignes_brutes  = []
        entree_courante = None

        # ── Étape 1 : Convertir le PDF en images ─────────────────────────────
        # first_page / last_page utilisent la numérotation 1-based de pdf2image
        print(f"Conversion des pages de traduction francaise du PDF en images {self.DPI} DPI…")
        pages_images: list[Image.Image] = convert_from_path(
            str(self.pdf_path),
            dpi=self.DPI,
            first_page=11,
            last_page=77,
        )
        print(f"  {len(pages_images)} pages converties.")

        # ── Étape 2 : OCR page par page ───────────────────────────────────────
        for num_page, image in enumerate(pages_images, start=11):
            largeur, hauteur = image.size

            # Découpage en deux colonnes égales (coordonnées en pixels)
            col_gauche_img = image.crop((0,           0, largeur // 2, hauteur))
            col_droite_img = image.crop((largeur // 2, 0, largeur,     hauteur))

            for col_img in [col_gauche_img, col_droite_img]:
                # OCR sur la colonne — langue française pour le dictionnaire
                texte_col = pytesseract.image_to_string(
                    col_img,
                    lang="fra",
                    config=self.TESSERACT_CONFIG,
                )

                if not texte_col.strip():
                    continue

                lignes = texte_col.split("\n")
                for ligne in lignes:
                    ligne = ligne.strip()
                    if not ligne:
                        continue

                    # Ignorer les numéros de page et en-têtes récurrents
                    if ligne.isdigit() or ligne.lower().startswith("sara languages lexicon"):
                        continue

                    # Détection des tags de dialectes dans la ligne
                    contient_lang = bool(re.search(lang_regex, ligne))

                    if contient_lang:
                        if entree_courante is not None:
                            correspondances = list(re.finditer(lang_regex, ligne))
                            for i, corr in enumerate(correspondances):
                                lang_code = corr.group(1)
                                start_idx = corr.end()
                                end_idx   = (
                                    correspondances[i + 1].start()
                                    if i + 1 < len(correspondances)
                                    else len(ligne)
                                )
                                mot = ligne[start_idx:end_idx].strip()
                                if mot:
                                    mot = self.clean_ocr_text(mot)
                                    if mot:
                                        entree_courante[lang_code] = mot
                    else:
                        # Nouvelle entrée (terme pivot en français)
                        if entree_courante is not None and len(entree_courante) > 1:
                            lignes_brutes.append(entree_courante)
                        entree_courante = {"french_term": ligne}

        # Enregistrement de la dernière entrée du dictionnaire
        if entree_courante is not None and len(entree_courante) > 1:
            lignes_brutes.append(entree_courante)

        return lignes_brutes

    # ─────────────────────────────────────────────────────────────────────────
    # Méthode 2 : Transformation au format LARGE
    # ─────────────────────────────────────────────────────────────────────────
    def to_wide(self) -> pd.DataFrame:
        """
        Transforme les données extraites en format large (un terme par ligne,
        une colonne par dialecte).
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
        print(f"  [Export] sara_wide.csv → {chemin_sortie}")

        return df_wide

    # ─────────────────────────────────────────────────────────────────────────
    # Méthode 3 : Transformation au format LONG
    # ─────────────────────────────────────────────────────────────────────────
    def to_long(self) -> pd.DataFrame:
        """
        Transforme les données en format long (paires de traduction empilées,
        une paire par ligne).
        """
        format_long_data = []

        for ligne in self.lignes_brutes:
            terme_francais = ligne.get("french_term")
            if not terme_francais:
                continue

            for langue_codee, nom_langue in SARA_LANGS.items():
                mot = ligne.get(langue_codee)
                if pd.notna(mot) and str(mot).strip():
                    format_long_data.append({
                        "source_lang": "French",
                        "source_word": terme_francais,
                        "target_lang": nom_langue,
                        "target_word": str(mot).strip(),
                    })

        df_long = pd.DataFrame(
            format_long_data,
            columns=["source_lang", "source_word", "target_lang", "target_word"],
        )

        chemin_sortie = self.output_dir / "sara_long.csv"
        df_long.to_csv(chemin_sortie, index=False, encoding="utf-8-sig")
        print(f"  [Export] sara_long.csv → {chemin_sortie}")

        return df_long

    # ─────────────────────────────────────────────────────────────────────────
    # Méthode 4 : Fichiers individuels par dialecte
    # ─────────────────────────────────────────────────────────────────────────
    def to_dialect_pairs(self) -> dict[str, pd.DataFrame]:
        """
        Produit un fichier CSV à deux colonnes pour chaque dialecte :
        <nom_dialecte> | français
        """
        pairs_dir = self.output_dir / "pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)

        dialecte_dfs = {}

        for langue_codee, nom_langue in SARA_LANGS.items():
            pairs_data = []

            for ligne in self.lignes_brutes:
                terme_francais = ligne.get("french_term")
                mot = ligne.get(langue_codee)

                if terme_francais and pd.notna(mot) and str(mot).strip():
                    pairs_data.append({
                        nom_langue.lower(): str(mot).strip(),
                        "français":         terme_francais,
                    })

            df_dialect = pd.DataFrame(
                pairs_data, columns=[nom_langue.lower(), "français"]
            )

            nom_fichier   = f"{nom_langue.lower()}_french.csv"
            chemin_sortie = pairs_dir / nom_fichier
            df_dialect.to_csv(chemin_sortie, index=False, encoding="utf-8-sig")

            dialecte_dfs[nom_langue] = df_dialect

        print(f"  [Export] {len(dialecte_dfs)} fichiers dialecte → {pairs_dir}/")
        return dialecte_dfs

    # ─────────────────────────────────────────────────────────────────────────
    # Méthode principale d'orchestration
    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        """
        Exécute l'intégralité du pipeline OCR → Parsing → Export CSV.
        """
        print("=" * 60)
        print("  EXTRACTION DU LEXIQUE SARA — Pipeline OCR")
        print("=" * 60)

        self.lignes_brutes = self.extract()
        print(f"{len(self.lignes_brutes)} entrées du dictionnaire chargées.")

        df_wide = self.to_wide()
        print(f"Format large : {df_wide.shape[0]} termes × {df_wide.shape[1]} colonnes.")

        df_long = self.to_long()
        print(f"Format long  : {len(df_long)} paires de traduction.")

        dialect_dfs = self.to_dialect_pairs()
        print(f"{len(dialect_dfs)} fichiers individuels par dialecte.")

        print("=" * 60)
        print("  PIPELINE TERMINÉ AVEC SUCCÈS")
        print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    extractor = SaraExtractor(
        pdf_path="/home/broly/TextMining_BLK/data/raw/SaraLanguagesLexicon.pdf",
        output_dir="/home/broly/TextMining_BLK/data/",
    )
    extractor.run()