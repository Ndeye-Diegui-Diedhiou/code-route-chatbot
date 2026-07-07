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

    # 1. Reformulation de la question pour optimiser la recherche sémantique (Vocabulaire juridique)
    search_query = req.question
    try:
        rewrite_completion = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Tu es un expert du code de la route sénégalais. Ton but est de transformer la question familière de l'utilisateur en une phrase exacte que l'on trouverait dans le texte de loi (le décret). Par exemple, 'doubler' devient 'dépassement', 'vitesse en ville' devient 'vitesse en agglomération'. Fais une seule courte phrase reprenant les termes légaux exacts de la question."},
                {"role": "user", "content": req.question}
            ],
            temperature=0.1,
            max_tokens=25,
        )
        search_query = rewrite_completion.choices[0].message.content.strip().replace('"', '')
    except Exception:
        pass

    # 2. Recherche dans ChromaDB avec les DEUX requêtes pour maximiser les chances
    results = collection.query(
        query_texts=[search_query, req.question],
        n_results=3
    )
    
    # Fusion des résultats des deux requêtes pour éviter les doublons
    unique_docs = {}
    for j in range(len(results['documents'])):
        for i, doc_id in enumerate(results['ids'][j]):
            if doc_id not in unique_docs:
                unique_docs[doc_id] = {
                    'text': results['documents'][j][i],
                    'metadata': results['metadatas'][j][i]
                }
                
    if not unique_docs:
        return AnswerResponse(
            answer="Je ne trouve pas d'article correspondant dans le Code de la route.",
            articles=[]
        )

    # On a retiré le filtre strict sur la distance car l'embedding Default est parfois imprécis
    # (distance > 0.8 bloquait de bons résultats)

    # 3. Construction du contexte
    context_parts = []
    articles_list = []

    # Prendre les 5 premiers documents uniques max
    for doc in list(unique_docs.values())[:5]:
        text = doc['text']
        meta = doc['metadata']

        article_number = meta.get('article_number', 'INCONNU')
        page = meta.get('page', 0)

        context_parts.append(f"--- {article_number} (Page {page}) ---\n{text}\n")
        articles_list.append(Article(article_number=article_number, page=page, text=text))

    context_str = "\n".join(context_parts)

    # 4. Génération avec Llama 3.3 70B via Groq
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
