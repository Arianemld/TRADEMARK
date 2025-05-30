# -*- coding: utf-8 -*-
"""
read_pdf_en.py: Extraction de champs structurés depuis des PDF de décisions de marques en anglais.
Utilise un modèle BERT anglais pour extraire via QA, gère le GPU, parallélise le traitement,
affiche une barre de progression et liste les champs manquants.
Champs extraits : decision_number, decision_date, applicant, opponent, classes_nice, motif_opposition, resultat_decision.
"""
import os
import re
import json
import logging
import csv
import warnings
from pathlib import Path
from datetime import datetime
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

# --- Initialisation du modèle BERT anglais en mode QA ---
MODEL_NAME = "deepset/roberta-base-squad2"  # Excellent modèle pour QA en anglais
device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
qa_pipeline = pipeline(
    task="question-answering",
    model=model.to('cuda' if device == 0 else 'cpu'),
    tokenizer=tokenizer,
    device=device
)

# --- Questions cibles en anglais ---
QUESTIONS = {
    'decision_number':    "What is the decision number?",
    'decision_date':      "What is the date of the decision?",
    'applicant':          "What is the name and number of the applicant's trademark?",
    'opponent':           "What is the name and number of the opponent?",
    'classes_nice':       "What are the NICE classes mentioned?",
    'motif_opposition':   "What is the ground for opposition?",
    'resultat_decision':  "What is the result of the decision?"
}

# Seuil minimal de confiance pour accepter une réponse
SCORE_THRESHOLD = 0.15

# --- Patterns regex pour extraction de fallback ---
FALLBACK_PATTERNS = {
    'decision_number': [
        r'(?:Decision|DECISION)\s*(?:No\.?|Number|#)\s*(\d+[-/]?\d*)',
        r'Case\s*(?:No\.?|Number|#)\s*(\d+[-/]?\d*)',
        r'Opposition\s*(?:No\.?|Number|#)\s*(\d+[-/]?\d*)'
    ],
    'decision_date': [
        r'(?:dated?|Date[d]?)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        r'(?:dated?|Date[d]?)\s*:?\s*(\d{1,2}\s+\w+\s+\d{4})',
        r'(?:on|On)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        r'(?:on|On)\s+(\d{1,2}\s+\w+\s+\d{4})'
    ],
    'applicant': [
        r'(?:Applicant|APPLICANT|Proprietor|PROPRIETOR)\s*:?\s*([^\n]+)',
        r'(?:Owner|OWNER)\s*:?\s*([^\n]+)',
        r'(?:Holder|HOLDER)\s*:?\s*([^\n]+)'
    ],
    'opponent': [
        r'(?:Opponent|OPPONENT)\s*:?\s*([^\n]+)',
        r'(?:Opposer|OPPOSER)\s*:?\s*([^\n]+)',
        r'(?:Opposing Party|OPPOSING PARTY)\s*:?\s*([^\n]+)'
    ],
    'classes_nice': [
        r'(?:class|Class|CLASS)(?:es)?\s*(?:NICE\s*)?(\d{1,2}(?:\s*,\s*\d{1,2})*)',
        r'(?:in|for)\s+(?:class|Class)(?:es)?\s*(\d{1,2}(?:\s*,\s*\d{1,2})*)',
        r'Nice\s+(?:class|Class)(?:es)?\s*(\d{1,2}(?:\s*,\s*\d{1,2})*)'
    ],
    'motif_opposition': [
        r'(?:grounds?|GROUNDS?)\s*(?:of|for)\s*(?:opposition|OPPOSITION)\s*:?\s*([^\n]{20,200})',
        r'(?:based on|BASED ON)\s*:?\s*([^\n]{20,200})',
        r'(?:opposition|OPPOSITION)\s+(?:is|was)\s+(?:based|founded)\s+(?:on|upon)\s*:?\s*([^\n]{20,200})'
    ],
    'resultat_decision': [
        r'(?:DECIDES?|ORDERS?|JUDGMENT|RULING)\s*:?\s*([^\n]{20,200})',
        r'(?:opposition|OPPOSITION)\s+(?:is|was)\s+(dismissed|upheld|rejected|allowed|granted|refused|sustained)',
        r'(?:The|THE)\s+(?:Board|BOARD|Office|OFFICE)\s+(dismisses?|upholds?|rejects?|allows?|grants?|refuses?)\s+(?:the\s+)?(?:opposition|OPPOSITION)'
    ]
}

# --- Mots-clés pour validation ---
VALIDATION_KEYWORDS = {
    'resultat_decision': ['dismissed', 'upheld', 'rejected', 'allowed', 'granted', 'refused', 'sustained', 'overruled', 'partially', 'entirely'],
    'motif_opposition': ['likelihood of confusion', 'similarity', 'identical', 'prior rights', 'bad faith', 'reputation', 'distinctive character', 'unfair advantage']
}

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


def extract_section(text, pattern, max_length):
    """Extrait une section spécifique du texte basée sur un pattern."""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        start = match.start()
        return text[start:start + max_length]
    return ""


def smart_chunk_text(text, max_length=2000, overlap=200):
    """Découpage intelligent qui priorise les sections clés pour documents anglais."""
    # Sections prioritaires
    sections = {
        'header': text[:3000],  # Début du document
        'decision': extract_section(text, r'DECISION|JUDGMENT|RULING|ORDERS?', 2000),
        'grounds': extract_section(text, r'GROUNDS?|REASONS?|OPPOSITION GROUNDS?', 2000),
        'facts': extract_section(text, r'FACTS|BACKGROUND|PROCEEDINGS', 2000),
        'conclusion': extract_section(text, r'CONCLUSION|ORDERS?|DECIDES?|FOR THESE REASONS?', 2000)
    }
    
    chunks = []
    # Ajouter les sections non vides
    for section_name, section_text in sections.items():
        if section_text and len(section_text.strip()) > 50:
            chunks.append(section_text)
    
    # Si pas assez de chunks, découper le texte complet
    if len(chunks) < 3:
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


def validate_and_clean_answer(field: str, answer: str) -> str:
    """Valide et nettoie les réponses selon le type de champ."""
    if not answer:
        return ''
    
    answer = answer.strip()
    
    if field == 'decision_number':
        # Extraire uniquement le numéro
        match = re.search(r'(\d+[-/]?\d*)', answer)
        return match.group(1) if match else answer
    
    elif field == 'decision_date':
        # Normaliser les formats de dates
        date_patterns = [
            (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', lambda m: f"{m.group(1)}/{m.group(2)}/{m.group(3)}"),
            (r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', 
             lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}"),
            (r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s+(\d{4})',
             lambda m: f"{m.group(2)} {m.group(1)} {m.group(3)}")
        ]
        for pattern, formatter in date_patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                return formatter(match)
        return answer
    
    elif field == 'classes_nice':
        # Extraire et valider les numéros de classes (1-45)
        classes = re.findall(r'\d+', answer)
        valid_classes = [c for c in classes if 1 <= int(c) <= 45]
        return ', '.join(sorted(set(valid_classes), key=int)) if valid_classes else answer
    
    elif field == 'motif_opposition':
        # Limiter la longueur et nettoyer
        answer = re.sub(r'\s+', ' ', answer)  # Normaliser les espaces
        if len(answer) > 500:
            answer = answer[:497] + "..."
        return answer
    
    elif field == 'resultat_decision':
        # Extraire le résultat principal
        answer = re.sub(r'\s+', ' ', answer)
        if len(answer) > 200:
            # Chercher la phrase clé
            for keyword in VALIDATION_KEYWORDS['resultat_decision']:
                if keyword in answer.lower():
                    start = answer.lower().find(keyword)
                    return answer[max(0, start-50):start+150]
            answer = answer[:197] + "..."
        return answer
    
    return answer


def extract_field_via_qa(field: str, text: str):
    """Extrait un champ via QA sur tous les chunks et agrège la meilleure réponse."""
    best_answer, best_score = '', 0.0
    question = QUESTIONS[field]
    
    chunks = smart_chunk_text(text)
    
    for chunk in chunks:
        answer, score = answer_question(question, chunk)
        if score > best_score:
            best_answer, best_score = answer, score
    
    # Nettoyer et valider la réponse
    if best_score >= SCORE_THRESHOLD and best_answer:
        cleaned_answer = validate_and_clean_answer(field, best_answer)
        return cleaned_answer, True
    
    return '', False


def extract_field_fallback(field: str, text: str) -> str:
    """Extraction par regex si le QA échoue."""
    if field not in FALLBACK_PATTERNS:
        return ''
    
    for pattern in FALLBACK_PATTERNS[field]:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            result = match.group(1).strip()
            # Valider avec les mots-clés si applicable
            if field in VALIDATION_KEYWORDS:
                if any(keyword in result.lower() for keyword in VALIDATION_KEYWORDS[field]):
                    return validate_and_clean_answer(field, result)
            else:
                return validate_and_clean_answer(field, result)
    
    return ''


def extract_with_heuristics(field: str, text: str, qa_answer: str, qa_success: bool):
    """Combine QA avec des heuristiques et fallback pour améliorer l'extraction."""
    # Si QA a réussi avec un bon score, utiliser sa réponse
    if qa_success and qa_answer:
        return qa_answer
    
    # Sinon, essayer le fallback regex
    fallback_answer = extract_field_fallback(field, text)
    
    # Si on a les deux réponses, choisir la meilleure
    if qa_answer and fallback_answer:
        # Préférer la réponse qui contient des mots-clés validés
        if field in VALIDATION_KEYWORDS:
            qa_valid = any(kw in qa_answer.lower() for kw in VALIDATION_KEYWORDS[field])
            fb_valid = any(kw in fallback_answer.lower() for kw in VALIDATION_KEYWORDS[field])
            if fb_valid and not qa_valid:
                return fallback_answer
        
        # Pour les numéros, préférer le plus court/simple
        if field in ['decision_number', 'classes_nice']:
            if len(fallback_answer) < len(qa_answer):
                return fallback_answer
    
    # Retourner la meilleure réponse disponible
    return qa_answer or fallback_answer or ''


# --- Traitement d'un PDF avec QA et suivi des champs manquants ---
def process_pdf(pdf_path_str):
    """Traite un seul PDF et extrait tous les champs."""
    pdf_path = Path(pdf_path_str)
    logging.info(f"Traitement de {pdf_path.name}")
    
    text = extract_full_text(pdf_path)
    if not text:
        logging.error(f"Aucun texte extrait de {pdf_path.name}")
        return {
            'filename': pdf_path.name,
            **{field: '' for field in QUESTIONS},
            'missing_fields': list(QUESTIONS.keys())
        }
    
    data = {'filename': pdf_path.name}
    missing = []
    
    for field in QUESTIONS:
        # Essayer d'abord l'extraction QA
        qa_answer, qa_success = extract_field_via_qa(field, text)
        
        # Combiner avec heuristiques et fallback
        final_answer = extract_with_heuristics(field, text, qa_answer, qa_success)
        
        data[field] = final_answer
        if not final_answer:
            missing.append(field)
    
    data['missing_fields'] = missing
    
    if missing:
        logging.warning(f"Champs manquants pour {pdf_path.name}: {missing}")
    else:
        logging.info(f"✓ Tous les champs extraits pour {pdf_path.name}")
    
    return data


# --- Processus de dossier, parallélisation, progression et export CSV ---
def process_folder(folder_path, output_csv='trademark_decisions_extracted.csv', max_workers=None):
    """Traite tous les PDFs d'un dossier et exporte en CSV."""
    folder = Path(folder_path)
    pdf_files = [str(p) for p in folder.glob('*.pdf')]
    
    if not pdf_files:
        logging.error(f"Aucun fichier PDF trouvé dans {folder_path}")
        return
    
    logging.info(f"Trouvé {len(pdf_files)} fichiers PDF à traiter")
    
    results = []
    
    # Déterminer le nombre de workers
    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)
    
    logging.info(f"Parallélisation sur {max_workers} workers...")
    
    # Traitement parallèle avec barre de progression
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pdf_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="PDFs traités"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                pdf_name = Path(futures[future]).name
                logging.error(f"Erreur lors du traitement de {pdf_name}: {e}")
                # Ajouter un résultat vide pour ce PDF
                results.append({
                    'filename': pdf_name,
                    **{field: '' for field in QUESTIONS},
                    'missing_fields': list(QUESTIONS.keys())
                })
    
    # Trier les résultats par nom de fichier
    results.sort(key=lambda x: x['filename'])
    
    # Statistiques
    total_files = len(results)
    complete_files = sum(1 for r in results if not r['missing_fields'])
    
    logging.info(f"\n--- Statistiques ---")
    logging.info(f"Fichiers traités: {total_files}")
    logging.info(f"Extractions complètes: {complete_files} ({complete_files/total_files*100:.1f}%)")
    
    # Statistiques par champ
    field_stats = {field: sum(1 for r in results if r.get(field)) for field in QUESTIONS}
    for field, count in field_stats.items():
        logging.info(f"{field}: {count}/{total_files} ({count/total_files*100:.1f}%)")
    
    # Sauvegarde CSV
    keys = ['filename'] + list(QUESTIONS.keys()) + ['missing_fields']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        
        for row in results:
            # Convertir la liste missing_fields en string
            row_serialized = row.copy()
            row_serialized['missing_fields'] = ';'.join(row.get('missing_fields', []))
            writer.writerow(row_serialized)
    
    logging.info(f"\n✅ Résultats sauvegardés dans {output_csv}")
    
    # Sauvegarder aussi un rapport JSON détaillé
    report_path = output_csv.replace('.csv', '_report.json')
    report = {
        'processing_date': datetime.now().isoformat(),
        'total_files': total_files,
        'complete_extractions': complete_files,
        'field_statistics': field_stats,
        'model_used': MODEL_NAME,
        'results': results
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logging.info(f"📊 Rapport détaillé sauvegardé dans {report_path}")


# --- Point d'entrée principal ---
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extraction de données depuis des PDFs de décisions de marques en anglais"
    )
    parser.add_argument('folder', help='Dossier contenant les fichiers PDF')
    parser.add_argument('--output', default='trademark_decisions_extracted.csv', 
                       help='Fichier CSV de sortie (défaut: trademark_decisions_extracted.csv)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Nombre de processus parallèles (défaut: auto)')
    
    args = parser.parse_args()
    
    # Vérifier que le dossier existe
    if not Path(args.folder).exists():
        logging.error(f"Le dossier {args.folder} n'existe pas")
        exit(1)
    
    # Lancer le traitement
    process_folder(args.folder, args.output, args.workers)