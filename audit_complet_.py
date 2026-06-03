import pdfplumber
from collections import Counter

pdf_path = "/home/broly/TextMining_BILAKE_Tchaa_Mèwè_Angelo/data/raw/SaraLanguagesLexicon.pdf"

# Ensemble pour stocker tous les caractères uniques du PDF
caracteres_trouves = Counter()

print("Analyse des 66 pages en cours...")
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            caracteres_found = list(text)
            caracteres_trouves.update(caracteres_found)

# Définir ce qu'on considère comme un caractère "normal" (lettres françaises, chiffres, ponctuation de base)
lettres_normales = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,()=:- \n\t'’\""

# Filtrer pour ne garder que les caractères suspects / Legacy Font
anomalies_globales = {char: count for char, count in caracteres_trouves.items() if char not in lettres_normales}

print("\n--- RAPPORT D'AUDIT DES CARACTÈRES ---")
print(f"Nombre de caractères suspects uniques trouvés : {len(anomalies_globales)}")
print("\nVoici la liste des caractères à vérifier (et leur fréquence d'apparition) :")
for char, count in sorted(anomalies_globales.items(), key=lambda x: x[1], reverse=True):
    # Affichage sécurisé pour les caractères invisibles ou sauts
    char_visuel = repr(char) if char.isspace() else char
    print(f"  Caractère : {char_visuel}  -> Apparaît {count} fois dans le PDF")