import re
import faiss
import pymupdf
import streamlit as st

from sentence_transformers import SentenceTransformer
from llama_cpp import Llama


# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="ResearchLens",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 ResearchLens")
st.write("Upload a research paper and ask questions about it.")


# ==================================================
# LOAD MODELS
# ==================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():

    return Llama(
        model_path="models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        n_ctx=2048,
        verbose=False
    )


embedding_model = load_embedding_model()
llm = load_llm()


# ==================================================
# KEYWORD SEARCH
# ==================================================

def keyword_search(chunks, query):

    query_lower = query.lower()

    # --------------------------------------------------
    # FIND TECHNICAL TERMS
    # Examples: SW420, DHT22, MQ135, ESP32
    # --------------------------------------------------

    technical_terms = re.findall(
        r"\b(?:dht\d+|mq\d+|sw\d+|esp\d+|pir)\b",
        query_lower
    )

    # --------------------------------------------------
    # EXACT TECHNICAL TERM SEARCH
    # --------------------------------------------------

    if technical_terms:

        exact_matches = []

        for chunk in chunks:

            text_lower = chunk["text"].lower()

            score = 0

            for term in technical_terms:

                if term in text_lower:

                    score += 100

            if score > 0:

                exact_matches.append(
                    (score, chunk)
                )

        exact_matches.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return [
            chunk
            for score, chunk in exact_matches[:5]
        ]

    # --------------------------------------------------
    # NORMAL KEYWORD SEARCH
    # --------------------------------------------------

    stop_words = {
        "what", "are", "is", "the", "a", "an",
        "of", "in", "on", "for", "to", "and",
        "or", "used", "use", "does", "do",
        "how", "why", "which", "what's"
    }

    query_words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        query_lower
    )

    important_words = [
        word
        for word in query_words
        if word not in stop_words
    ]

    phrases = []

    for i in range(
        len(important_words) - 1
    ):

        phrase = (
            important_words[i]
            + " "
            + important_words[i + 1]
        )

        phrases.append(
            phrase
        )

    matches = []

    for chunk in chunks:

        text = chunk["text"].lower()

        score = 0

        # Exact phrase match
        for phrase in phrases:

            if phrase in text:

                score += 5

        # Individual word match
        for word in important_words:

            if word in text:

                score += 1

        # --------------------------------------------------
        # PRIORITIZE PROPOSED SYSTEM
        # --------------------------------------------------

        proposed_keywords = [

            "proposed system",

            "proposed smart grain storage system",

            "hardware design",

            "working of smart grain storage system",

            "working of the smart grain storage system",

            "proposed design"
        ]

        for keyword in proposed_keywords:

            if keyword in text:

                score += 10

        # --------------------------------------------------
        # REDUCE LITERATURE SURVEY PRIORITY
        # --------------------------------------------------

        literature_keywords = [

            "literature survey",

            "in paper [",

            "paper [1]",

            "paper [2]",

            "paper [3]",

            "previous study",

            "existing study"
        ]

        for keyword in literature_keywords:

            if keyword in text:

                score -= 8

        if score > 0:

            matches.append(
                (score, chunk)
            )

    matches.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        chunk
        for score, chunk in matches[:5]
    ]


# ==================================================
# FIND TECHNICAL TERMS IN TEXT
# ==================================================

def find_technical_terms(text):

    patterns = [

        r"\bDHT\d+\b",

        r"\bMQ\d+\b",

        r"\bSW\d+\b",

        r"\bPIR\b",

        r"\bESP\d+\b",

        r"\b[A-Z]{2,}\d+\b"
    ]

    terms = []

    for pattern in patterns:

        found = re.findall(
            pattern,
            text
        )

        for term in found:

            if term not in terms:

                terms.append(
                    term
                )

    return terms


# ==================================================
# PDF UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "📄 Upload your research paper",
    type=["pdf"]
)


if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # --------------------------------------------------
    # READ PDF
    # --------------------------------------------------

    pdf_bytes = uploaded_file.getvalue()

    pdf = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    chunks = []

    chunk_size = 600

    overlap = 100

    step = chunk_size - overlap


    # --------------------------------------------------
    # CREATE CHUNKS
    # --------------------------------------------------

    for page_number, page in enumerate(
        pdf,
        start=1
    ):

        text = page.get_text().strip()

        if not text:

            continue

        for i in range(
            0,
            len(text),
            step
        ):

            chunk_text = text[
                i:i + chunk_size
            ].strip()

            if chunk_text:

                chunks.append({

                    "text": chunk_text,

                    "page": page_number

                })


    st.info(
        f"📚 Document processed: {len(chunks)} chunks"
    )


    # ==================================================
    # CREATE EMBEDDINGS
    # ==================================================

    texts = [

        chunk["text"]

        for chunk in chunks

    ]

    embeddings = embedding_model.encode(

        texts,

        convert_to_numpy=True

    )


    dimension = embeddings.shape[1]


    # ==================================================
    # CREATE FAISS INDEX
    # ==================================================

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings.astype("float32")
    )


    # ==================================================
    # QUESTION
    # ==================================================

    question = st.text_input(
        "🔍 Ask a question about your paper:"
    )


    if question:

        # ==================================================
        # SEMANTIC SEARCH
        # ==================================================

        question_embedding = embedding_model.encode(

            [question],

            convert_to_numpy=True

        )


        distances, results = index.search(

            question_embedding.astype(
                "float32"
            ),

            5

        )


        # ==================================================
        # KEYWORD SEARCH
        # ==================================================

        keyword_chunks = keyword_search(

            chunks,

            question

        )


        # ==================================================
        # COMBINE RESULTS
        # ==================================================

        selected_chunks = []

        seen_text = set()


        # Keyword results first
        for chunk in keyword_chunks:

            chunk_text = chunk["text"].strip()

            normalized_text = " ".join(

                chunk_text.lower().split()

            )

            if normalized_text in seen_text:

                continue

            seen_text.add(
                normalized_text
            )

            selected_chunks.append(
                chunk
            )


        # Semantic results second
        for result_index in results[0]:

            chunk = chunks[result_index]

            chunk_text = chunk["text"].strip()

            normalized_text = " ".join(

                chunk_text.lower().split()

            )

            if normalized_text in seen_text:

                continue

            seen_text.add(
                normalized_text
            )

            selected_chunks.append(
                chunk
            )


        # Keep maximum 6 chunks
        selected_chunks = selected_chunks[:6]


        # ==================================================
        # BUILD CONTEXT
        # ==================================================

        context_parts = []

        source_pages = []

        technical_terms = []


        for chunk in selected_chunks:

            chunk_text = chunk["text"]


            context_parts.append(
                chunk_text
            )


            source_pages.append(
                chunk["page"]
            )


            found_terms = find_technical_terms(
                chunk_text
            )


            for term in found_terms:

                if term not in technical_terms:

                    technical_terms.append(
                        term
                    )


        context = (
            "\n\n--- PDF SOURCE ---\n\n"
            .join(context_parts)
        )


        # ==================================================
        # LLM PROMPT
        # ==================================================

        prompt = f"""<|im_start|>system
You are ResearchLens, a document question-answering assistant.

Answer the user's question using ONLY the PDF context.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not guess.
3. Do not invent information.
4. Use the exact terminology from the PDF.
5. If the PDF gives a specific sensor name, model number, component name, or technical term, use that exact name.
6. Do not replace a specific technical name with a different name.
7. Do not create names that are not present in the PDF.
8. Do not mix information from different studies.
9. If the question asks about the proposed system, prefer information from:
   - Proposed System
   - Hardware Design
   - Proposed Design
   - Working of Smart Grain Storage System
10. Do NOT take hardware or sensor names from the Literature Survey if they belong to another referenced paper.
11. If the question contains an exact technical component name such as SW420, DHT22, MQ135, ESP32, or PIR, answer using the PDF text that specifically discusses that component.
12. Do not confuse nearby components.
13. For example, if the PDF says:
   SW420 Vibration Sensor -> detects movement or vibration
   DC Cooling Fan -> maintains air circulation
   then do not say SW420 is used for the DC Cooling Fan.
14. If several items are listed, include the relevant unique items.
15. Do not repeat the same item.
16. Ignore unrelated information.
17. If the answer is not supported by the PDF context, reply exactly:
"I could not find the answer in the document."
18. Keep the answer short and clear.

TECHNICAL TERMS FOUND IN RETRIEVED PDF TEXT:
{", ".join(technical_terms)}

PDF CONTEXT:
{context}

USER QUESTION:
{question}

<|im_end|>
<|im_start|>assistant
"""


        # ==================================================
        # GENERATE ANSWER
        # ==================================================

        with st.spinner(
            "ResearchLens is thinking..."
        ):

            output = llm(

                prompt,

                max_tokens=200,

                temperature=0.1

            )


        answer = output[
            "choices"
        ][0][
            "text"
        ].strip()


        # ==================================================
        # DISPLAY ANSWER
        # ==================================================

        st.subheader(
            "Answer"
        )

        st.write(
            answer
        )


        # ==================================================
        # RETRIEVED TEXT
        # ==================================================

        with st.expander(
            "🔍 View retrieved PDF text"
        ):

            for i, chunk in enumerate(

                selected_chunks,

                start=1

            ):

                st.write(

                    f"### Retrieved chunk {i} — "
                    f"Page {chunk['page']}"

                )

                st.write(
                    chunk["text"]
                )


        # ==================================================
        # SOURCE PAGES
        # ==================================================

        source_pages = sorted(
            set(source_pages)
        )


        st.info(

            "📄 Source pages: "

            + ", ".join(

                str(page)

                for page in source_pages

            )

        )