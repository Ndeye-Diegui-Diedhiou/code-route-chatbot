import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from groq import Groq
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Configuration Groq (Llama 3.3 70B — open source, gratuit)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY non trouvée dans le fichier .env")

groq_client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"

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

    # 3. Génération avec Llama 3.3 70B via Groq
    try:
        chat_completion = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant expert sur le Code de la route du Sénégal. "
                        "Tu réponds UNIQUEMENT en te basant sur le contexte fourni par le décret officiel. "
                        "Si le contexte ne contient pas la réponse, dis-le clairement. "
                        "Tu DOIS citer les numéros d'articles utilisés (ex: 'Selon l\\'ARTICLE 42...'). "
                        "Réponds en français, de manière claire et structurée."
                    )
                },
                {
                    "role": "user",
                    "content": f"Question : {req.question}\n\nContexte extrait du Code de la route :\n{context_str}"
                }
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        answer = chat_completion.choices[0].message.content
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            answer = "⚠️ Limite de requêtes atteinte. Veuillez réessayer dans quelques instants."
        else:
            answer = f"Erreur lors de la génération : {err}"

    return AnswerResponse(
        answer=answer,
        articles=articles_list
    )

# Servir le frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
