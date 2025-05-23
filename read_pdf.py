# Pour éviter l'affichage des logs du pdfplumber
import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

import os
import pdfplumber
import re
import csv

folder = "decisions"
results = []
non_trouves = []

for filename in os.listdir(folder):
    if filename.endswith(".pdf"):
        path = os.path.join(folder, filename)
        with pdfplumber.open(path) as pdf:
            # 1. Lire la première page pour extraire le numéro de décision
            first_page = pdf.pages[0]
            text = first_page.extract_text()

            match = re.search(r"O\s*/?\s*-?\s*\d{3,4}\s*/?\s*-?\s*\d{2}", text, re.IGNORECASE)
            if match:
                decision_number = re.sub(r"\s+", "", match.group()).replace("-", "")
            else:
                decision_number = "Non trouvé"
                non_trouves.append(filename)

            # 2. Parcourir toutes les pages pour construire le texte complet
            full_text = ""
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    full_text += content + "\n"

            # 3. Recherche des classes NICE dans tout le texte
            class_mentions = re.findall(r"(?:class(?:es)?)\s+[0-9,\sand&]+", full_text, re.IGNORECASE)

            if class_mentions:
                all_classes = []
                for mention in class_mentions:
                    all_classes.extend(re.findall(r"\d{1,2}", mention))
                classes = ", ".join(sorted(set(all_classes), key=int))
            else:
                classes = "Non trouvée"

            # 4. Recherche de l'applicant (nettoyage intégré)
            applicant_match = re.search(
                r"(?:APPLICATION NO\.|INTERNATIONAL REGISTRATION\s+NO\.)[^\n]*\n(?:DESIGNATING THE UK\s*)?\n?BY\s+([^\n]+)",
                full_text, re.IGNORECASE)
            if applicant_match:
                applicant = applicant_match.group(1).strip()
                applicant = re.sub(r"^(BY|TO REGISTER|IN THE NAME OF)\s*", "", applicant, flags=re.IGNORECASE).strip()
            else:
                applicant = "Non trouvé"


            # 5. Recherche de l'opponent (nettoyage intégré)
            opponent_match = re.search(
                r"OPPOSITION(?: THERETO)?\s+UNDER NO\..*?\n(?:BY|IN THE NAME OF)?\s*([^\n]+)",
                full_text, re.IGNORECASE)
            if opponent_match:
                opponent = opponent_match.group(1).strip()
                opponent = re.sub(r"^(BY|AND|IN THE NAME OF)\s*", "", opponent, flags=re.IGNORECASE).strip()
            else:
                opponent = "Non trouvé"

            # 6. Sauvegarde du résultat pour ce fichier
            results.append({
                "filename": filename,
                "decision_number": decision_number,
                "applicant": applicant,
                "opponent": opponent,
                "classes": classes
            })

# 7. Écriture des résultats dans un CSV
with open("decision_numbers.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "decision_number", "applicant", "opponent", "classes"])
    writer.writeheader()
    writer.writerows(results)

print("✅ Extraction terminée. Résultats enregistrés dans 'decision_numbers.csv'.")

# 8. Affichage des fichiers où le numéro de décision est manquant
if non_trouves:
    print("\n❌ Aucun numéro trouvé dans les fichiers suivants :")
    for name in non_trouves:
        print(" -", name)
