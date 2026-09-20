import pymupdf

pdf = pymupdf.open("documents/paper.pdf")

chunk_size = 1000

chunks = []

for page_number, page in enumerate(pdf, start=1):

    text = page.get_text().strip()

    for i in range(0, len(text), chunk_size):

        chunk_text = text[i:i + chunk_size]

        chunks.append({
            "text": chunk_text,
            "page": page_number
        })

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:5]):

    print("\nCHUNK:", i + 1)
    print("PAGE:", chunk["page"])
    print(chunk["text"])
    print("-------------------------")