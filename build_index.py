import os
import pickle
import faiss
import pymupdf

from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Read PDF
# -----------------------------

pdf_path = "documents/paper.pdf"

pdf = pymupdf.open(pdf_path)

chunks = []

chunk_size = 600
overlap = 100

for page_number, page in enumerate(pdf, start=1):

    text = page.get_text().strip()

    for i in range(0, len(text), chunk_size - overlap):

        chunk_text = text[i:i + chunk_size]

        if chunk_text.strip():

            chunks.append({
                "text": chunk_text,
                "page": page_number
            })


print("Total chunks:", len(chunks))


# -----------------------------
# 2. Create embeddings
# -----------------------------

print("Creating embeddings...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in chunks]

embeddings = embedding_model.encode(
    texts,
    convert_to_numpy=True
)


# -----------------------------
# 3. Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(
    embeddings.astype("float32")
)

print("FAISS vectors:", index.ntotal)


# -----------------------------
# 4. Create data folder
# -----------------------------

os.makedirs("data", exist_ok=True)


# -----------------------------
# 5. Save FAISS index
# -----------------------------

faiss.write_index(
    index,
    "data/researchlens.index"
)


# -----------------------------
# 6. Save chunks
# -----------------------------

with open("data/chunks.pkl", "wb") as file:

    pickle.dump(chunks, file)


print("\nIndex created successfully!")

print("Saved:")
print("data/researchlens.index")
print("data/chunks.pkl")