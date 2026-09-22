import re
from pathlib import Path

import pymupdf


def clean_text(text: str) -> str:
    """Clean obvious PDF extraction noise while preserving content."""

    # Normalize whitespace while preserving paragraph structure.
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)

    # Remove repeated blank lines.
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def extract_pages(pdf_path: str, source: str, document_name: str) -> list[dict]:
    """Extract each PDF page as a structured document."""

    pdf = pymupdf.open(pdf_path)

    documents = []

    for page_number, page in enumerate(pdf, start=1):
        raw_text = page.get_text()
        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            continue

        documents.append(
            {
                "text": cleaned_text,
                "metadata": {
                    "source": source,
                    "document": document_name,
                    "page": page_number,
                },
            }
        )

    pdf.close()

    return documents


def save_documents(documents: list[dict], output_path: str) -> None:
    """Save structured documents as a readable text file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        for document in documents:
            metadata = document["metadata"]

            file.write(
                f"\n{'=' * 80}\n"
                f"SOURCE: {metadata['source']}\n"
                f"DOCUMENT: {metadata['document']}\n"
                f"PAGE: {metadata['page']}\n"
                f"{'=' * 80}\n\n"
            )

            file.write(document["text"])
            file.write("\n")


if __name__ == "__main__":

    constitution_documents = extract_pages(
        "data/raw/constitution/constitution.pdf",
        source="constitution",
        document_name="Constitution of India",
    )

    ncert_documents = extract_pages(
        "data/raw/ncert/NCERTPolity.pdf",
        source="ncert",
        document_name="NCERT Polity",
    )

    all_documents = constitution_documents + ncert_documents

    save_documents(
        all_documents,
        "data/processed/prepared_documents.txt",
    )

    print(f"Constitution pages: {len(constitution_documents)}")
    print(f"NCERT pages: {len(ncert_documents)}")
    print(f"Total pages: {len(all_documents)}")
    print("Saved to: data/processed/prepared_documents.txt")