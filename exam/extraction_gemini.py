"""
Partie A — Extraction du lexique Sara via vision IA (Gemini 2.5 Flash).

Problème racine : polices héritées sans table ToUnicode → extraction native impossible.
Solution : envoyer chaque page comme image à Gemini Vision qui comprend la mise en
page du dictionnaire et retourne directement du JSON structuré.
"""

import json
import os
import time
from pathlib import Path

from google import genai
import pandas as pd
from pdf2image import convert_from_path

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
    Extrait le lexique Sara via vision IA Gemini.
    Pipeline : PDF → images 300 DPI → Gemini Vision → JSON → CSV (wide/long/pairs).
    """

    def __init__(self, pdf_path: str, output_dir: str = "data/", first_page: int = 11, last_page: int = 77):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.first_page = first_page
        self.last_page = last_page
        self.raw_rows: list[dict] = []

    def extract(self) -> list[dict]:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Variable d'environnement GEMINI_API_KEY non définie.\n"
                "  export GEMINI_API_KEY='votre_cle_ici'"
            )
        client = genai.Client(api_key=api_key)

        print(f"Conversion pages {self.first_page}–{self.last_page} en images 300 DPI…")
        images = convert_from_path(
            str(self.pdf_path),
            dpi=300,
            first_page=self.first_page,
            last_page=self.last_page,
        )
        print(f"  {len(images)} pages converties.")

        prompt = """
Tu es un expert en transcription linguistique. Analyse l'image de cette page de
dictionnaire Sara-Français et extrait les entrées au format JSON.

RÈGLES DE TRANSCRIPTION STRICTES :
- Respecte EXACTEMENT les glyphes imprimés (ɓ, ɗ, ɛ, ɔ, ɨ, ə, ḭ̀, ə̰̀, etc.)
- Ne normalise JAMAIS les caractères
- Ne confonds pas : ə ≠ e, ɨ ≠ i, ɛ ≠ e, ɔ ≠ o, ɓ ≠ b, ɗ ≠ d
- Conserve les accents et signes diacritiques (tilde souscrit  ̰ , tons)

Pour chaque entrée, trouve le terme français (pivot), puis les traductions
dans les dialectes marqués par leurs codes (Beb=, Bd=, Db=, Gor=, Gu=,
KbN=, Kbb=, Lk=, Mb=, Mo=, Nar=, Ngb=, Sr=, NgT=).

Retourne UNIQUEMENT un tableau JSON valide :
[
  {"french_term": "...", "Beb": "...", "Bd": "...", ...},
  ...
]

Si un dialecte n'a pas de traduction, mets une chaîne vide "".
"""
        raw_data = []

        for idx, img in enumerate(images):
            print(f"  Page {self.first_page + idx} / {self.last_page}…", end=" ", flush=True)

            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[img, prompt],
                        config={
                            "response_mime_type": "application/json",
                            "temperature": 0.0,
                        },
                    )
                    page_data = json.loads(response.text)

                    if isinstance(page_data, list):
                        raw_data.extend(page_data)
                    elif isinstance(page_data, dict) and "entries" in page_data:
                        raw_data.extend(page_data["entries"])

                    print(f"{len(page_data if isinstance(page_data, list) else page_data.get('entries', []))} entrées")
                    time.sleep(3.5)
                    break

                except Exception as e:
                    err = str(e).lower()
                    if "429" in err or "quota" in err or "resource exhausted" in err:
                        wait = 10.0 * (attempt + 1)
                        print(f"quotas ({wait}s)")
                    else:
                        wait = 5.0
                        print(f"erreur: {e}")
                    time.sleep(wait)

            else:
                print(f"  Page {self.first_page + idx}: échec après 3 tentatives, ignorée.")

        return raw_data

    def to_wide(self) -> pd.DataFrame:
        df = pd.DataFrame(self.raw_rows)
        for lang in SARA_LANGS:
            if lang not in df.columns:
                df[lang] = ""
        ordered = ["french_term"] + list(SARA_LANGS.keys())
        df = df.reindex(columns=ordered).fillna("")
        out = self.output_dir / "sara_wide.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        return df

    def to_long(self) -> pd.DataFrame:
        df_wide = self.to_wide()
        df_long = df_wide.melt(
            id_vars=["french_term"],
            value_vars=list(SARA_LANGS.keys()),
            var_name="target_lang_code",
            value_name="target_word",
        )
        df_long = df_long[df_long["target_word"].str.strip() != ""]
        df_long["source_lang"] = "French"
        df_long["source_word"] = df_long["french_term"]
        df_long["target_lang"] = df_long["target_lang_code"].map(SARA_LANGS)
        df_long = df_long[["source_lang", "source_word", "target_lang", "target_word"]]
        out = self.output_dir / "sara_long.csv"
        df_long.to_csv(out, index=False, encoding="utf-8-sig")
        return df_long

    def to_dialect_pairs(self) -> dict[str, pd.DataFrame]:
        df_wide = self.to_wide()
        pairs_dir = self.output_dir / "pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)
        dialect_dfs = {}
        for code, full_name in SARA_LANGS.items():
            df_d = df_wide[["french_term", code]].copy()
            df_d = df_d[df_d[code].str.strip() != ""]
            lang_lower = full_name.lower()
            df_d.columns = ["français", lang_lower]
            df_d = df_d[[lang_lower, "français"]]
            fpath = pairs_dir / f"{lang_lower}_french.csv"
            df_d.to_csv(fpath, index=False, encoding="utf-8-sig")
            dialect_dfs[code] = df_d
        return dialect_dfs

    def run(self):
        print("=" * 60)
        print("  EXTRACTION — Pipeline Gemini Vision")
        print("=" * 60)
        self.raw_rows = self.extract()
        print(f"\n{len(self.raw_rows)} entrées extraites.")

        dw = self.to_wide()
        print(f"wide : {dw.shape[0]} termes × {dw.shape[1]} colonnes")

        dl = self.to_long()
        print(f"long  : {len(dl)} paires")

        dd = self.to_dialect_pairs()
        print(f"{len(dd)} fichiers dialecte dans data/pairs/")
        print("=" * 60)


if __name__ == "__main__":
    extractor = SaraExtractor(
        pdf_path="/home/broly/TextMining_BILAKE_Tchaa_Mèwè_Angelo/data/raw/SaraLanguagesLexicon.pdf",
        output_dir="/home/broly/TextMining_BILAKE_Tchaa_Mèwè_Angelo/data/",
    )
    extractor.run()
