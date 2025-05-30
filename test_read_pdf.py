# -*- coding: utf-8 -*-
"""
read_pdf.py: Extraction de champs structurés depuis des PDF de décisions de marques.
Utilise un modèle BERT (CamemBERT) en local pour extraire via QA, gère le GPU, parallélise le traitement,
affiche une barre de progression et liste les champs manquants.
Champs extraits : decision_number, applicant, opponent, classes_nice, motif_opposition, resultat_decision.
"""
import os
import re
import logging
import csv
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pdfplumber
from tqdm import tqdm
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# --- Configuration et suppression des warnings inutiles ---
warnings.filterwarnings(
    "ignore",
    message="CropBox missing from /Page, defaulting to MediaBox"
)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Initialisation du modèle BERT (CamemBERT) en mode QA ---
MODEL_NAME = "etalab-ia/camembert-base-squadFR-fquad-piaf"
device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
qa_pipeline = pipeline(
    task="question-answering",
    model=model.to('cuda' if device == 0 else 'cpu'),
    tokenizer=tokenizer,
    device=device
)

# --- Questions cibles ---
QUESTIONS = {
    'decision_number':    "Quel est le numéro de la décision ?",  
    'applicant':          "Quel est le nom et numéro de la marque déposée par l'applicant ?",  
    'opponent':           "Quel est le nom et numéro de l'opposant ?",  
    'classes_nice':       "Quelles sont les classes NICE mentionnées ?",  
    'motif_opposition':   "Quel est le motif de l'opposition ?",  
    'resultat_decision':  "Quel est le résultat de la décision ?"
}

# Seuil minimal de confiance pour accepter une réponse
SCORE_THRESHOLD = 0.1

# --- Fonctions utilitaires ---
def extract_full_text(pdf_path):
    """Extrait et concatène le texte de toutes les pages du PDF."""
    texts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
    except Exception as e:
        logging.error(f"Erreur ouverture {pdf_path.name}: {e}")
    return "\n".join(texts)


def chunk_text(text, max_length=2000, overlap=200):
    """Découpe le texte en chunks de longueur max_length avec overlap chars."""
    if len(text) <= max_length:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        chunks.append(text[start:end])
        start += max_length - overlap
    return chunks


def answer_question(question: str, context: str):
    """Appelle le pipeline QA et renvoie answer, score."""
    try:
        out = qa_pipeline(question=question, context=context)
        return out.get('answer', '').strip(), out.get('score', 0.0)
    except Exception as e:
        logging.debug(f"QA error: {e}")
        return '', 0.0


def extract_field_via_qa(field: str, text: str):
    """Extrait un champ via QA sur tous les chunks et agrège la meilleure réponse."""
    best_answer, best_score = '', 0.0
    question = QUESTIONS[field]
    for chunk in chunk_text(text):
        answer, score = answer_question(question, chunk)
        if score > best_score:
            best_answer, best_score = answer, score
    if best_score < SCORE_THRESHOLD or not best_answer:
        return '', False
    return best_answer, True

# --- Traitement d'un PDF avec QA et suivi des champs manquants ---
def process_pdf(pdf_path_str):
    pdf_path = Path(pdf_path_str)
    text = extract_full_text(pdf_path)
    data = { 'filename': pdf_path.name }
    missing = []

    for field in QUESTIONS:
        ans, ok = extract_field_via_qa(field, text)
        data[field] = ans
        if not ok:
            missing.append(field)
    data['missing_fields'] = missing
    if missing:
        logging.warning(f"Manquants {pdf_path.name}: {missing}")
    return data

# --- Processus de dossier, parallélisation, progression et export CSV ---

def process_folder(folder_path, output_csv='decisions_extraites_bert.csv'):
    folder = Path(folder_path)
    pdf_files = [str(p) for p in folder.glob('*.pdf')]
    results = []

    max_workers = min(4, os.cpu_count() or 1)
    logging.info(f"Parallélisation sur {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pdf_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="PDF traités"):
            try:
                results.append(future.result())
            except Exception as e:
                logging.error(f"Erreur {futures[future]}: {e}")

    keys = ['filename'] + list(QUESTIONS.keys()) + ['missing_fields']
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            row_serialized = {k: ';'.join(row[k]) if isinstance(row[k], list) else row[k] for k in keys}
            writer.writerow(row_serialized)
    logging.info(f"✅ Sauvegarde vers {output_csv}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Extraction de décisions via BERT QA")
    parser.add_argument('folder', help='Dossier des PDF')
    parser.add_argument('--output', default='decisions_extraites_bert.csv', help='Fichier CSV de sortie')
    args = parser.parse_args()
    process_folder(args.folder, args.output)
