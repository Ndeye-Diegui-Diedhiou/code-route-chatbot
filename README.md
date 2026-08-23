<div align="center">
  <h1>🚦 Code de la Route Sénégalais<br>Coach IA (RAG)</h1>
  <p>
    <strong>Un assistant intelligent pour réviser et interroger le Code de la Route sénégalais.</strong>
  </p>
  <p>
    <a href="https://code-route-chatbot.onrender.com"><strong>Accéder à la démo en ligne »</strong></a>
  </p>

  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.9+-blue.svg" />
    <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi" />
    <img alt="Groq" src="https://img.shields.io/badge/Groq-Llama%203.3-orange" />
    <img alt="ChromaDB" src="https://img.shields.io/badge/ChromaDB-Vector%20Search-blueviolet" />
  </p>
</div>

<br />

Ce projet est un chatbot pédagogique basé sur le Code de la route sénégalais, utilisant la technologie RAG (**Retrieval-Augmented Generation**). Il découpe un document PDF officiel par article, stocke les fragments dans une base vectorielle, et utilise **Llama 3.3 70B** (via Groq) pour générer des réponses précises en citant les articles de loi.

---

## ✨ Fonctionnalités

- **Recherche Sémantique Avancée** : Traduit automatiquement les requêtes "familières" en vocabulaire juridique (ex: _doubler_ → _dépassement_) pour une meilleure précision.
- **Réponses Sourcées** : Chaque réponse cite explicitement l'Article ou l'Annexe du décret officiel du Sénégal.
- **Ton Pédagogique** : Agit comme un moniteur d'auto-école pour vous aider à réviser. Si la loi ne précise pas un détail (ex: distance de stationnement près d'un virage), l'IA vous l'explique naturellement.
- **Interface Premium** : Design épuré, officiel et rapide.

---

## 🛠️ Architecture Technique

- **Extraction PDF** : `pdfplumber` avec découpage sémantique intelligent basé sur les mots-clés "ARTICLE" et "ANNEXE".
- **Base Vectorielle** : `ChromaDB` (stockage local avec `DefaultEmbeddingFunction` ONNX, optimisé pour les faibles ressources mémoire).
- **Backend API** : `FastAPI`.
- **LLM / Génération IA** : `Llama-3.3-70b-versatile` via l'API **Groq** (extrêmement rapide et gratuit).

---

## 🚀 Installation & Lancement (Local)

### 1. Prérequis
- Python 3.9+
- Une clé API Groq gratuite (créable sur [console.groq.com](https://console.groq.com/keys))

### 2. Configuration
Clonez le dépôt, puis installez les dépendances via un environnement virtuel :

```bash
# 1. Créer l'environnement virtuel
python -m venv venv

# 2. Activer l'environnement
source venv/bin/activate      # Sur Linux / macOS
.\venv\Scripts\activate       # Sur Windows

# 3. Installer les librairies
pip install -r requirements.txt
```

Créez un fichier `.env` à la racine du projet et ajoutez votre clé API :
```env
GROQ_API_KEY=votre_cle_groq_ici
```

### 3. Ingestion du Document (Base de Données)
Le décret PDF officiel (`decret-code-route-senegal.pdf`) doit être placé dans `data/documents/`. Pour générer la base vectorielle locale :
```bash
python -m ingestion.ingest
```
*(Cela va extraire les ~766 articles et annexes dans un dossier `chroma_db`.)*

### 4. Démarrage de l'API & Interface
Lancez le backend FastAPI via Uvicorn :
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
L'API et l'interface Web (fichiers statiques) seront servies sur le même port.
👉 **Ouvrez simplement [http://127.0.0.1:8000](http://127.0.0.1:8000) dans votre navigateur !**

---

## ☁️ Déploiement (Render)

Ce projet est prêt à être déployé sur des plateformes comme Render.com en tant que "Web Service" utilisant Python 3.
- **Build Command** : `pip install -r requirements.txt && python -m ingestion.ingest`
- **Start Command** : `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables** : N'oubliez pas d'ajouter `GROQ_API_KEY`.

---
<div align="center">
  <i>Fait avec ❤️ pour aider à l'apprentissage des règles de sécurité routière au Sénégal.</i>
</div>
