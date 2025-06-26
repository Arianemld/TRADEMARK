import os
import re
import json
import fitz  # PyMuPDF
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Pour ne pas afficher les logs de pdfminer
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Initialisation du modèle NER
MODEL_NAME = "Jean-Baptiste/roberta-large-ner-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

folder = "decisions"
results = []
non_trouves = []

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text("text") for page in doc)

def extract_entities(text):
    return [e for e in ner_pipeline(text) if e.get('entity_group') != 'MISC']

def clean_name(name):
    return re.sub(r"^(BY|AND|IN THE NAME OF|TO REGISTER|APPLICATION NO\.|DESIGNATING THE UK)\s*", "", name, flags=re.IGNORECASE).strip(" :")

def extract_parties_info(text, entities):
    applicant = opponent = "Non trouvé"

    app_match = re.search(r"IN THE MATTER OF .*?\n(?:BY|IN THE NAME OF)\s+([^\n]+)", text, re.IGNORECASE)
    opp_match = re.search(r"OPPOSITION(?: THERETO)?\s+UNDER NO\..*?\n(?:BY|IN THE NAME OF)?\s*([^\n]+)", text, re.IGNORECASE)

    if app_match:
        applicant = clean_name(app_match.group(1))
    if opp_match:
        opponent = clean_name(opp_match.group(1))

    orgs = [e['word'] for e in entities if e['entity_group'] in ('ORG', 'PER')]
    if applicant == "Non trouvé" and orgs:
        applicant = clean_name(orgs[0])
    if opponent == "Non trouvé" and len(orgs) > 1:
        opponent = clean_name(orgs[1])

    return applicant, opponent

def extract_decision_number(text):
    match = re.search(r"O\s*/?\s*-?\s*\d{3,4}\s*/?\s*-?\s*\d{2}", text, re.IGNORECASE)
    if match:
        return re.sub(r"\s+", "", match.group()).replace("-", "")
    return "Non trouvé"

def extract_classes(text):
    class_mentions = re.findall(r"(?:class(?:es)?)\s+[0-9,\sand&]+", text, re.IGNORECASE)
    all_classes = []
    for mention in class_mentions:
        all_classes.extend(re.findall(r"\d{1,2}", mention))
    return ", ".join(sorted(set(all_classes), key=int)) if all_classes else "Non trouvée"


def extract_comparison_sections(text):
    section_titles = [
        "Comparison of the marks",
        "Comparison of the goods and services",
        "Likelihood of confusion"
    ]

    extracted = {}
    for title in section_titles:
        pattern = rf"{title[:10]}.*?(?=\n[A-Z]|\Z)"

        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted[title] = match.group().strip()

    return extracted



# Traitement des fichiers PDF dans le dossier



def process_folder():
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)
            text = extract_text_from_pdf(path)

            decision_number = extract_decision_number(text)
            if decision_number == "Non trouvé":
                non_trouves.append(filename)

            classes = extract_classes(text)
            entities = extract_entities(text)
            applicant, opponent = extract_parties_info(text, entities)
            comparison_sections = extract_comparison_sections(text)


            results.append({
                "filename": filename,
                "decision_number": decision_number,
                "applicant": applicant,
                "opponent": opponent,
                "classes": classes,
                "comparisons": comparison_sections
            })

    with open("decision_numbers.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    if non_trouves:
        print("\n❌ Aucun numéro trouvé dans les fichiers suivants :")
        for name in non_trouves:
            print(" -", name)

if __name__ == "__main__":
    process_folder()
    print("\n✅ Extraction terminée. Résultats enregistrés dans 'decision_numbers.json'.")
