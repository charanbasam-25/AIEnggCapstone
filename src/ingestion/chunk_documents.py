from pathlib import Path
import re


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks."""

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def parse_prepared_documents(file_path: str) -> list[dict]:
    """Read prepared_documents.txt and recover page metadata."""

    text = Path(file_path).read_text(encoding="utf-8")

    # Each page starts with a SOURCE/DOCUMENT/PAGE metadata block.
    pattern = re.compile(
        r"={80}\s*"
        r"SOURCE:\s*(.*?)\s*"
        r"DOCUMENT:\s*(.*?)\s*"
        r"PAGE:\s*(\d+)\s*"
        r"={80}\s*"
        r"(.*?)(?=\n={80}\s*SOURCE:|\Z)",
        re.DOTALL,
    )

    documents = []

    for match in pattern.finditer(text):

        source = match.group(1).strip()
        document = match.group(2).strip()
        page = int(match.group(3))
        content = match.group(4).strip()

        if not content:
            continue

        documents.append(
            {
                "text": content,
                "metadata": {
                    "source": source,
                    "document": document,
                    "page": page,
                },
            }
        )

    return documents
def create_chunks(documents: list[dict]) -> list[dict]:
    """Create retrieval chunks while preserving source metadata."""

    chunks = []

    for document in documents:
        text_chunks = split_text(document["text"])

        for chunk_index, chunk_text in enumerate(text_chunks):

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        **document["metadata"],
                        "chunk_index": chunk_index,
                    },
                }
            )

    return chunks


def save_chunks(chunks: list[dict], output_path: str) -> None:
    """Save chunks for inspection."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:

        for index, chunk in enumerate(chunks):

            metadata = chunk["metadata"]

            file.write(
                f"\n{'=' * 80}\n"
                f"CHUNK: {index}\n"
                f"SOURCE: {metadata['source']}\n"
                f"DOCUMENT: {metadata['document']}\n"
                f"PAGE: {metadata['page']}\n"
                f"CHUNK INDEX: {metadata['chunk_index']}\n"
                f"{'=' * 80}\n\n"
            )

            file.write(chunk["text"])
            file.write("\n")


if __name__ == "__main__":

    documents = parse_prepared_documents(
        "data/processed/prepared_documents.txt"
    )

    chunks = create_chunks(documents)

    save_chunks(
        chunks,
        "data/processed/chunks.txt",
    )

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print("Saved to: data/processed/chunks.txt")