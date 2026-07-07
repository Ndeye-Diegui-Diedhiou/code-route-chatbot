import os
import re
import pdfplumber
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Configuration
PDF_PATH = os.path.join("data", "documents", "decret-code-route-senegal.pdf")
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "code_route"

def extract_text_and_chunk(pdf_path):
    print(f"Extraction du texte depuis {pdf_path}...")

    pattern = re.compile(r'(?=(?:ARTICLE\s+[A-Z0-9\-]+|ANNEXE\s+[A-Z0-9\-]+))', re.IGNORECASE)

    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages_text.append({"text": text, "page": i + 1})

    combined_text = ""
    page_markers = []

    current_idx = 0
    for p in pages_text:
        page_markers.append((current_idx, p["page"]))
        combined_text += p["text"] + "\n"
        current_idx = len(combined_text)

    raw_chunks = pattern.split(combined_text)

    chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        title_match = re.match(r'(ARTICLE\s+[A-Z0-9\-]+|ANNEXE\s+[A-Z0-9\-]+)', chunk, re.IGNORECASE)
        article_number = title_match.group(1).upper() if title_match else "INCONNU"

        if article_number == "INCONNU" and len(chunk) < 50:
            continue

        idx = combined_text.find(chunk)
        page_num = 1
        for start_idx, p_num in page_markers:
            if start_idx <= idx:
                page_num = p_num
            else:
                break

        chunks.append({
            "text": chunk,
            "metadata": {
                "article_number": article_number,
                "page": page_num
            },
            "id": f"chunk_{len(chunks)}"
        })

    print(f"{len(chunks)} articles/annexes extraits.")
    return chunks

def ingest_to_chroma(chunks):
    print("Initialisation de ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Utilisation de la fonction d'embedding légère ONNX (pas de PyTorch)
    emb_fn = DefaultEmbeddingFunction()

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except ValueError:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Création des embeddings et insertion dans ChromaDB ({len(chunks)} chunks)...")

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        ids = [c["id"] for c in batch]

        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} inséré.")

    print("Ingestion terminée avec succès.")

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"Erreur: Le fichier {PDF_PATH} n'existe pas.")
        exit(1)

    chunks = extract_text_and_chunk(PDF_PATH)
    if chunks:
        ingest_to_chroma(chunks)
    else:
        print("Aucun chunk généré.")
