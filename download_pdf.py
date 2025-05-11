import requests
from bs4 import BeautifulSoup
import os

# URL de la page des décisions
url = "https://www.ipo.gov.uk/t-challenge-decision-results.htm"

# En-têtes pour contourner l'erreur 403 (anti-bot)
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

# Faire la requête HTTP avec les headers
response = requests.get(url, headers=headers)
response.raise_for_status()  # Stoppe si la requête échoue

# Parser le HTML avec BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")

# Créer un dossier pour enregistrer les fichiers
os.makedirs("decisions", exist_ok=True)

# Récupérer tous les liens vers des PDF de décisions
pdf_links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    if href.endswith(".pdf") and "/t-challenge-decision-results/" in href:
        full_url = "https://www.ipo.gov.uk" + href
        pdf_links.append(full_url)

print(f"✅ {len(pdf_links)} fichiers PDF trouvés.")

# Télécharger les fichiers PDF
for link in pdf_links:
    filename = link.split("/")[-1]
    filepath = os.path.join("decisions", filename)
    if not os.path.exists(filepath):  # éviter les doublons
        pdf_response = requests.get(link, headers=headers)
        with open(filepath, "wb") as f:
            f.write(pdf_response.content)
        print(f"⬇️ Téléchargé : {filename}")
    else:
        print(f"✅ Déjà présent : {filename}")
