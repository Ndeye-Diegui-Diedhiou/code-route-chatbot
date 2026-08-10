import os
from datetime import datetime, timedelta, date
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import chromadb
from groq import Groq
from dotenv import load_dotenv

# Base de données
from sqlalchemy.orm import Session
from .database import engine, Base, get_db
from . import models

# Sécurité
from passlib.context import CryptContext
import jwt

# Chargement des variables d'environnement
load_dotenv()

# Création des tables de la base de données
try:
    Base.metadata.create_all(bind=engine)
    print("Base de données initialisée avec succès.")
except Exception as e:
    print(f"Erreur lors de la connexion à la base de données : {e}")

# Configuration de la sécurité JWT
SECRET_KEY = os.getenv("SECRET_KEY", "super_secret_code_route_key_2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Configuration Groq (Llama 3.3 70B)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY non trouvée dans le fichier .env")

groq_client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"

# Configuration ChromaDB
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "code_route"

import pathlib
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

# Forcer le cache du modèle ONNX
ONNXMiniLM_L6_V2.DOWNLOAD_PATH = pathlib.Path(os.getcwd()) / "chroma_onnx_cache" / "all-MiniLM-L6-v2"

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    emb_fn = ONNXMiniLM_L6_V2()
    collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
    print(f"ChromaDB chargée : {collection.count()} documents.")
except Exception as e:
    print(f"Attention: Impossible de charger la base ChromaDB. Erreur: {e}")
    collection = None

app = FastAPI(title="API Code Route Sénégal - Freemium")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modèles Pydantic ---
class UserCreate(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserResponse(BaseModel):
    email: str
    is_premium: bool
    questions_remaining: int

class QuestionRequest(BaseModel):
    question: str

class Article(BaseModel):
    article_number: str
    page: int
    text: str

class AnswerResponse(BaseModel):
    answer: str
    articles: list[Article]
    questions_remaining: int
    is_premium: bool


# --- Utilitaires de Sécurité ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Identifiants invalides",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        raise credentials_exception
    return user


# --- Routes d'Authentification ---
@app.post("/register", response_model=Token)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Cet email est déjà enregistré")
    
    hashed_pwd = get_password_hash(user.password)
    new_user = models.User(email=user.email, hashed_password=hashed_pwd)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token = create_access_token(data={"sub": new_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Email ou mot de passe incorrect")
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me", response_model=UserResponse)
def get_me(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Calcul des questions restantes aujourd'hui
    today = date.today()
    usage = db.query(models.DailyUsage).filter(
        models.DailyUsage.user_id == current_user.id,
        models.DailyUsage.date == today
    ).first()
    
    count = usage.question_count if usage else 0
    remaining = max(0, 5 - count) if not current_user.is_premium else 999
    
    return UserResponse(
        email=current_user.email,
        is_premium=current_user.is_premium,
        questions_remaining=remaining
    )

@app.post("/upgrade")
def upgrade_premium(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Simulation de paiement réussie (mockup)
    current_user.is_premium = True
    db.commit()
    return {"message": "Succès ! Vous êtes maintenant Premium."}


# --- Route Principale (RAG) ---
@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # --- Vérification du Quota ---
    today = date.today()
    usage = db.query(models.DailyUsage).filter(
        models.DailyUsage.user_id == current_user.id,
        models.DailyUsage.date == today
    ).first()
    
    if not usage:
        usage = models.DailyUsage(user_id=current_user.id, date=today, question_count=0)
        db.add(usage)
        db.commit()
        db.refresh(usage)
        
    if not current_user.is_premium and usage.question_count >= 5:
        raise HTTPException(status_code=403, detail="QUOTA_REACHED")
    
    # --- RAG Logic ---
    if not collection:
        return AnswerResponse(
            answer="Erreur: La base de données vectorielle n'est pas initialisée.",
            articles=[],
            questions_remaining=0,
            is_premium=current_user.is_premium
        )

    search_query = req.question
    try:
        rewrite_completion = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Tu es un expert du code de la route sénégalais. Transforme la question familière en une courte phrase exacte utilisant les termes légaux."},
                {"role": "user", "content": req.question}
            ],
            temperature=0.1,
            max_tokens=25,
        )
        search_query = rewrite_completion.choices[0].message.content.strip().replace('"', '')
    except Exception:
        pass

    results = collection.query(
        query_texts=[search_query, req.question],
        n_results=3
    )
    
    unique_docs = {}
    for j in range(len(results['documents'])):
        for i, doc_id in enumerate(results['ids'][j]):
            if doc_id not in unique_docs:
                unique_docs[doc_id] = {
                    'text': results['documents'][j][i],
                    'metadata': results['metadatas'][j][i]
                }
                
    if not unique_docs:
        # On ne débite pas de quota si on ne trouve rien du tout
        return AnswerResponse(
            answer="Je ne trouve pas d'article correspondant dans le Code de la route.",
            articles=[],
            questions_remaining=max(0, 5 - usage.question_count) if not current_user.is_premium else 999,
            is_premium=current_user.is_premium
        )

    context_parts = []
    articles_list = []
    for doc in list(unique_docs.values())[:5]:
        text = doc['text']
        meta = doc['metadata']
        article_number = meta.get('article_number', 'INCONNU')
        page = meta.get('page', 0)
        context_parts.append(f"--- {article_number} (Page {page}) ---\n{text}\n")
        articles_list.append(Article(article_number=article_number, page=page, text=text))
    
    context_str = "\n".join(context_parts)

    try:
        chat_completion = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un coach expert sur le Code de la route du Sénégal. "
                        "Réponds en t'appuyant strictement sur les extraits du décret officiel. "
                        "Cite toujours les numéros d'articles pertinents. "
                        "Sois pédagogique, clair et direct."
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
        
        # Débit du quota en cas de succès
        usage.question_count += 1
        db.commit()
        
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            answer = "⚠️ Limite de requêtes atteinte. Veuillez réessayer dans quelques instants."
        else:
            answer = f"Erreur lors de la génération : {err}"

    remaining = max(0, 5 - usage.question_count) if not current_user.is_premium else 999

    return AnswerResponse(
        answer=answer,
        articles=articles_list,
        questions_remaining=remaining,
        is_premium=current_user.is_premium
    )


# Servir le frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
