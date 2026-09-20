import pymupdf
from sentence_transformers import SentenceTransformer

# Open PDF
pdf = pymupdf.open("documents/paper.pdf")

# Store chunks
chunks = []

chunk_size = 1000

# Extract text and keep page number
for page_number, page in enumerate(pdf, start=1):

    text = page.get_text().strip()

    for i in range(0, len(text), chunk_size):

        chunk_text = text[i:i + chunk_size]

        chunks.append({
            "text": chunk_text,
            "page": page_number
        })


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get only the text from each chunk
texts = [chunk["text"] for chunk in chunks]

# Convert text into embeddings
embeddings = model.encode(texts)

print("Total chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)

print("\nFirst chunk:")
print(chunks[0]["text"])

print("\nFirst embedding:")
print(embeddings[0])