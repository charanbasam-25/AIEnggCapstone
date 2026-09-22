import pymupdf
from pathlib import Path


def extract_text_from_pdf(pdf_path: str, output_path: str) -> None:
    pdf = pymupdf.open(pdf_path)

    extracted_pages = []

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text()

        extracted_pages.append(
            f"\n--- PAGE {page_number} ---\n\n{text}"
        )

    full_text = "\n".join(extracted_pages)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(full_text, encoding="utf-8")

    print(f"PDF: {pdf_path}")
    print(f"Pages: {len(pdf)}")
    print(f"Characters extracted: {len(full_text)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    extract_text_from_pdf(
        "data/raw/constitution/constitution.pdf",
        "data/processed/constitution.txt",
    )

    extract_text_from_pdf(
        "data/raw/ncert/NCERTPolity.pdf",
        "data/processed/ncert_polity.txt",
    )