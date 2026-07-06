# Code de la Route Sénégalais - Chatbot RAG

Ce projet est un chatbot basé sur le Code de la route sénégalais, utilisant la technologie RAG (Retrieval-Augmented Generation). Il découpe un document PDF du code de la route par article, stocke les fragments dans une base vectorielle, et utilise Gemini pour générer des réponses précises en citant les articles.

## Prérequis
- Python 3.9+
- Clé API Gemini (configurable dans un fichier `.env`)

## Installation

1. Clonez ce dépôt ou placez-vous dans le répertoire du projet.
2. Créez un environnement virtuel (optionnel mais recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
4. Obtenez une clé API Gemini gratuite sur [Google AI Studio](https://aistudio.google.com/) et ajoutez-la au fichier `.env` à la racine :
   ```env
   GEMINI_API_KEY=votre_cle_api_ici
   ```

## Utilisation

### 1. Ingestion du document
Le document source (PDF) doit être placé dans `data/documents/decret-code-route-senegal.pdf`.
Lancez le script d'ingestion pour peupler la base vectorielle (ChromaDB) :
```bash
python -m ingestion.ingest
```

### 2. Démarrage de l'API
Lancez le backend FastAPI :
```bash
python -m uvicorn api.main:app --reload
```
L'API sera disponible sur `http://localhost:8000`.

### 3. Interface Web
Ouvrez simplement le fichier `frontend/index.html` dans votre navigateur Web pour interagir avec le Chatbot. Vous pourrez lui poser des questions (par ex: "Quelle est la limitation de vitesse en agglomération ?") et il vous répondra en citant les numéros d'articles.

## Architecture technique
- **Extraction PDF** : `pdfplumber` avec découpage sémantique par article.
- **Base Vectorielle** : `ChromaDB` (stockage local).
- **Embeddings** : Modèle gratuit `all-MiniLM-L6-v2` via `sentence-transformers`.
- **Backend API** : `FastAPI`.
- **Génération IA** : API Google Gemini (`gemini-2.0-flash`).
