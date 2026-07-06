import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Configuration Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY non trouvée dans le fichier .env")

genai.configure(api_key=GEMINI_API_KEY)
# On utilise le modèle gratuit recommandé
model = genai.GenerativeModel('gemini-2.0-flash')

# Configuration ChromaDB
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "code_route"

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
except Exception as e:
    print(f"Attention: Impossible de charger la base ChromaDB. Avez-vous lancé l'ingestion ? Erreur: {e}")
    collection = None

app = FastAPI(title="API Code Route Sénégal")

# Autoriser les requêtes CORS pour le frontend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class Article(BaseModel):
    article_number: str
    page: int
    text: str

class AnswerResponse(BaseModel):
    answer: str
    articles: list[Article]

@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    if not collection:
        return AnswerResponse(
            answer="Erreur: La base de données n'est pas initialisée. Veuillez lancer l'ingestion d'abord.",
            articles=[]
        )
        
    # 1. Recherche dans ChromaDB
    results = collection.query(
        query_texts=[req.question],
        n_results=4
    )
    
    if not results['documents'] or not results['documents'][0]:
        return AnswerResponse(
            answer="Je ne trouve pas d'article correspondant dans le Code de la route.",
            articles=[]
        )
        
    # Vérification du score de similarité (distance cosinus : plus c'est proche de 0, mieux c'est)
    # ChromaDB renvoie des distances. Si la distance est trop élevée, on rejette.
    distances = results['distances'][0]
    # On vérifie si la meilleure distance est pertinente. Le seuil dépend du modèle.
    # Pour all-MiniLM-L6-v2 en cosinus, < 0.7 est généralement pertinent.
    if distances[0] > 0.8: 
        return AnswerResponse(
            answer="Je ne trouve pas d'article correspondant dans le Code de la route.",
            articles=[]
        )

    # 2. Construction du contexte
    context_parts = []
    articles_list = []
    
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        
        article_number = meta.get('article_number', 'INCONNU')
        page = meta.get('page', 0)
        
        context_parts.append(f"--- {article_number} (Page {page}) ---\n{text}\n")
        articles_list.append(Article(article_number=article_number, page=page, text=text))
        
    context_str = "\n".join(context_parts)
    
    # 3. Prompt pour Gemini
    prompt = f"""Tu es un assistant expert sur le Code de la route du Sénégal.
On t'a posé la question suivante : "{req.question}"

Voici le contexte extrait directement du texte officiel du Code de la route. Tu dois répondre UNIQUEMENT en te basant sur ce contexte. 
Si le contexte ne contient pas la réponse, réponds "Je ne trouve pas d'article correspondant dans le Code de la route".

Tu DOIS CITER les numéros d'articles que tu utilises pour ta réponse (ex: "Selon l'ARTICLE 42...").

CONTEXTE :
{context_str}

RÉPONSE :"""

    # 4. Génération de la réponse
    try:
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"Erreur lors de la génération avec Gemini : {str(e)}"
        
    return AnswerResponse(
        answer=answer,
        articles=articles_list
    )
