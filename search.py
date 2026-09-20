import pymupdf
import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Read PDF
# -----------------------------

pdf = pymupdf.open("documents/paper.pdf")

chunks = []

chunk_size = 1000


# -----------------------------
# 2. Create chunks
# -----------------------------

for page_number, page in enumerate(pdf, start=1):

    text = page.get_text().strip()

    for i in range(0, len(text), chunk_size):

        chunk_text = text[i:i + chunk_size]

        chunks.append({
            "text": chunk_text,
            "page": page_number
        })


# -----------------------------
# 3. Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 4. Create embeddings
# -----------------------------

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    convert_to_numpy=True
)


# -----------------------------
# 5. Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings.astype("float32"))


print("Total chunks:", len(chunks))
print("FAISS vectors:", index.ntotal)


# -----------------------------
# 6. Ask a question
# -----------------------------

question = "What is the Internet of Things?"


# Convert question into embedding
question_embedding = model.encode(
    [question],
    convert_to_numpy=True
)


# -----------------------------
# 7. Search FAISS
# -----------------------------

distances, results = index.search(
    question_embedding.astype("float32"),
    3
)


# -----------------------------
# 8. Display results
# -----------------------------

print("\nQUESTION:")
print(question)

print("\nMOST RELEVANT CHUNKS:")

for i, result_index in enumerate(results[0]):

    print("\nRESULT:", i + 1)

    print("PAGE:", chunks[result_index]["page"])

    print("DISTANCE:", distances[0][i])

    print(chunks[result_index]["text"])

    print("-------------------------")