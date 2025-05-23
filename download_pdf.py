#request ne marche pas dans notre cas parce que le site utilise surement une protection du coup j'utilise cloudscraper qui est une bibliothéque qui permet de contourner la protection anti-bot de cloudflare
import cloudscraper
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

BASE_URL = "https://www.ipo.gov.uk"
START_PAGE = "https://www.ipo.gov.uk/t-challenge-decision-results/t-challenge-decision-results-gen.htm?YearFrom=2024&YearTo=2024"

scraper = cloudscraper.create_scraper()

def get_pdf_link_from_decision_page(url):
   
    try:
        resp = scraper.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a['href']
            if href.lower().endswith(".pdf"):
                return urljoin(BASE_URL, href)
        return None
    except Exception as e:
        print(f"Erreur en accédant à {url} : {e}")
        return None

def main():
    # Récupérer la page principale
    resp = scraper.get(START_PAGE)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Extraire les liens vers pages individuelles de décisions
    decision_links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag['href']
        if "t-challenge-decision-results-bl" in href:
            full_url = urljoin(BASE_URL, href)
            decision_links.append(full_url)

    print(f"Nombre de pages individuelles trouvées : {len(decision_links)}")

    # Pour chaque page individuelle, récupérer le lien PDF et l'afficher
    pdf_links_found = []
    for idx, decision_url in enumerate(decision_links, 1):
        print(f"[{idx}/{len(decision_links)}] Traitement de {decision_url}")

        pdf_url = get_pdf_link_from_decision_page(decision_url)
        if pdf_url is None:
            print("Pas de PDF trouvé, passage au suivant.")
            continue

        print(f"Lien PDF trouvé : {pdf_url}")
        pdf_links_found.append(pdf_url)

        time.sleep(0.5)  # Pause légère sinon j'ai des erreurs 

    print(f"\nTotal de PDF trouvés : {len(pdf_links_found)}")

if __name__ == "__main__":
    main()
