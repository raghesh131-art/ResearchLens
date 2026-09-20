import pymupdf
import faiss

from sentence_transformers import SentenceTransformer
from llama_cpp import Llama


# -----------------------------
# 1. Read PDF
# -----------------------------

pdf = pymupdf.open("documents/paper.pdf")

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

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

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
# 4. Load local Qwen model
# -----------------------------

model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    verbose=False
)


# -----------------------------
# 5. Ask question
# -----------------------------

question = input("\nEnter your question: ")


# -----------------------------
# 6. Search FAISS
# -----------------------------

question_embedding = embedding_model.encode(
    [question],
    convert_to_numpy=True
)

distances, results = index.search(
    question_embedding.astype("float32"),
    1
)

print("\nRETRIEVED CHUNKS:")

for rank, result_index in enumerate(results[0], start=1):
    chunk = chunks[result_index]

    print("\nRESULT:", rank)
    print("PAGE:", chunk["page"])
    print(chunk["text"][:800])



# -----------------------------
# 7. Build context
# -----------------------------

context = ""
pages = []

for rank, result_index in enumerate(results[0], start=1):
    chunk = chunks[result_index]

    context += f"\n--- Source {rank} (Page {chunk['page']}) ---\n"
    context += chunk["text"] + "\n"

    pages.append(chunk["page"])


# -----------------------------
# 8. Create RAG prompt
# -----------------------------

prompt = f"""<|im_start|>system
You are ResearchLens, a helpful research assistant.

Answer the user's question using ONLY the provided PDF context.

If the answer is not present in the context, say:
"I could not find the answer in the document."

Give a short and clear answer.

Use the most directly relevant information from the context.
Do not focus on specific examples unless the question asks for an example.

IMPORTANT:
Do not use your own knowledge.
Do not guess.
Do not answer from memory.
If the PDF context does not contain the answer, reply exactly:
"I could not find the answer in the document."

<|im_end|>
<|im_start|>user

PDF CONTEXT:
{context}

QUESTION:
{question}

<|im_end|>
<|im_start|>assistant
"""


# -----------------------------
# 9. Generate answer
# -----------------------------

output = llm(
    prompt,
    max_tokens=150,
    temperature=0.2
)

answer = output["choices"][0]["text"].strip()


# -----------------------------
# 10. Display result
# -----------------------------

print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(answer)

print("\nSOURCE PAGES:")

for page in sorted(set(pages)):
    print("Page", page)