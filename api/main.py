import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Configuration Gemini (nouveau SDK google.genai)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY non trouvée dans le fichier .env")

client_genai = genai.Client(api_key=GEMINI_API_KEY)

# Configuration ChromaDB — embedding léger ONNX (pas de PyTorch)
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "code_route"

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    emb_fn = DefaultEmbeddingFunction()
    collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
    print(f"ChromaDB chargée : {collection.count()} documents.")
except Exception as e:
    print(f"Attention: Impossible de charger la base ChromaDB. Erreur: {e}")
    collection = None

app = FastAPI(title="API Code Route Sénégal")

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

    distances = results['distances'][0]
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

    # 4. Génération de la réponse avec le nouveau SDK
    try:
        response = client_genai.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1024,
            )
        )
        answer = response.text
    except Exception as e:
        answer = f"Erreur lors de la génération avec Gemini : {str(e)}"

    return AnswerResponse(
        answer=answer,
        articles=articles_list
    )

# Servir le frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
