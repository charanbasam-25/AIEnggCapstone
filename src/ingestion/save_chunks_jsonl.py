import json
from pathlib import Path

from chunk_documents import (
    create_chunks,
    parse_prepared_documents,
)


def save_chunks_jsonl(
    chunks: list[dict],
    output_path: str,
) -> None:
    """Save each chunk as one JSON object per line."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        for chunk_id, chunk in enumerate(chunks):

            record = {
                "id": chunk_id,
                "text": chunk["text"],
                "source": chunk["metadata"]["source"],
                "document": chunk["metadata"]["document"],
                "page": chunk["metadata"]["page"],
                "chunk_index": chunk["metadata"]["chunk_index"],
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":

    documents = parse_prepared_documents(
        "data/processed/prepared_documents.txt"
    )

    chunks = create_chunks(documents)

    save_chunks_jsonl(
        chunks,
        "data/processed/chunks.jsonl",
    )

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print("Saved to: data/processed/chunks.jsonl")