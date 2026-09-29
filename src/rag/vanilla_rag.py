import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from src.retrieval.retrieval_config import (
    RERANK_CANDIDATE_K,
    RETRIEVAL_TOP_K,
)
from src.retrieval.semantic_reranker import (
    SemanticReranker,
    load_chunks,
)

load_dotenv()


MODEL_NAME = "gpt-4o-mini"


class RAGAnswer(BaseModel):
    selected_option: str = Field(pattern="^[ABCD]$")
    explanation: str


class VanillaRAG:

    def __init__(self, chunks: list[dict]):

        # The baseline shares the verifier's retrieval depths on purpose.
        # System A vs System B is meant to isolate the verification layer;
        # if the two arms retrieved at different depths the comparison
        # would be measuring retrieval as well, which is the confound that
        # produced the k=3 defect. Importing both from one place makes
        # that impossible to break by editing one file.
        self.retriever = SemanticReranker(
            chunks,
            candidate_k=RERANK_CANDIDATE_K,
        )

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def answer(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> dict:

        # ---------------------------------------------
        # 1. Retrieve relevant evidence
        # ---------------------------------------------

        results = self.retriever.retrieve(
            query,
            top_k=top_k,
        )

        # ---------------------------------------------
        # 2. Build grounded context
        # ---------------------------------------------

        context_parts = []

        for rank, result in enumerate(
            results,
            start=1,
        ):
            context_parts.append(
                f"""
SOURCE {rank}
Document: {result["document"] if "document" in result else "Constitution of India"}
Source type: {result["source"]}
Page: {result["page"]}

{result["text"]}
"""
            )

        context = "\n".join(context_parts)

        # ---------------------------------------------
        # 3. Ask the LLM to answer from the evidence
        # ---------------------------------------------

        system_prompt = """
You are a source-grounded UPSC Indian Polity tutor.

Answer the user's question using ONLY the provided
source evidence.

Do not use outside knowledge.

If the evidence is insufficient to answer the question,
say that the provided sources are insufficient.

For every factual claim, cite the relevant source page
using this format:

[Constitution of India, p. X]

Give a clear and concise explanation.
"""

        user_prompt = f"""
Question:
{query}

Source Evidence:
{context}
"""

        response = self.client.beta.chat.completions.parse(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format=RAGAnswer,
        )

        parsed = response.choices[0].message.parsed

        if parsed is None:
            raise ValueError("Structured RAG response was not parsed")

        answer = parsed.explanation

        # ---------------------------------------------
        # 4. Return answer + retrieved evidence
        # ---------------------------------------------

        return {
            "query": query,
            "answer": answer,
            "selected_option": parsed.selected_option,
            "sources": [
                {
                    "source": result["source"],
                    "page": result["page"],
                    "score": result["reranker_score"],
                }
                for result in results
            ],
        }


if __name__ == "__main__":

    chunks = load_chunks(
        "data/processed/chunks.jsonl"
    )

    rag = VanillaRAG(chunks)

    query = "How is the President of India elected?"

    result = rag.answer(query)

    print("\n=== ANSWER ===\n")
    print(result["answer"])

    print("\n=== RETRIEVED SOURCES ===\n")

    for source in result["sources"]:
        print(
            f"{source['source']} "
            f"page {source['page']} "
            f"score={source['score']:.4f}"
        )