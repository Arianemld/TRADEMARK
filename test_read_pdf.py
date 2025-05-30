import os, re, csv, pdfplumber
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# 1. Modèle public
MODEL_NAME = "AgentPublic/camembert-base-squadFR-fquad-piaf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
qa        = pipeline("question-answering", model=model, tokenizer=tokenizer)

# 2. Sliding window QA
def extract_qa(question, context, window_size=512, stride=128):
    inputs = tokenizer(context, return_overflowing_tokens=True,
                       max_length=window_size, stride=stride,
                       truncation=True)
    best = {"score": 0.0, "answer": None}
    for i in range(len(inputs["input_ids"])):
        span = {
            "question": question,
            "context": tokenizer.decode(inputs["input_ids"][i])
        }
        res = qa(span)
        if res["score"] > best["score"]:
            best = res
    return best["answer"] or "Non détecté"

# 3. Regex fallback
def extract_regex(pattern, text, default="Non détecté"):
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else default

QUESTIONS = {
    "numero_decision":   "Quel est le numéro de la décision ?",
    "date_decision":     "Quelle est la date de décision ?",
    "nom_marques":       "Quels sont le(s) nom(s) et numéro(s) des marques concernées ?",
    "classes_nice":      "Quelles sont les classes NICE mentionnées ?",
    "motif_opposition":  "Quel est le motif de l'opposition ?",
    "resultat_decision": "Quel est le résultat de la décision ?"
}

REGEX_PATTERNS = {
    "numero_decision":   r"Décision\s*n°\s*([\w/-]+)",
    "date_decision":     r"\b\d{1,2}\s*(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s*\d{4}\b",
    "classes_nice":      r"classes?\s*:\s*([\d,\s]+)"
}

def extract_full_text(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            txt = p.extract_text()
            if txt: pages.append(txt)
    return "\n".join(pages)

def process_pdf(path):
    raw = extract_full_text(path)
    data = {"filename": Path(path).name}
    for field, question in QUESTIONS.items():
        # priorité QA si context < 100k caractères, sinon regex
        if len(raw) < 100_000:
            ans = extract_qa(question, raw)
        else:
            ans = extract_regex(REGEX_PATTERNS.get(field, "."), raw)
        data[field] = ans
    return data

def process_folder(folder, output_csv="decisions.csv"):
    rows = []
    for pdf in Path(folder).glob("*.pdf"):
        print(f"→ {pdf.name}")
        rows.append(process_pdf(pdf))
    keys = ["filename"] + list(QUESTIONS.keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print("✅ Fini :", output_csv)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="Dossier des PDF")
    p.add_argument("--output", default="decisions.csv")
    args = p.parse_args()
    process_folder(args.folder, args.output)
