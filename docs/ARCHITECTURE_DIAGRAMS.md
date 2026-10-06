# Overall system design: UPSC Polity AI Tutor

For the current learner app, see the [7 October practice architecture image](diagrams/practice_architecture.png)
and [project brief](PROJECT_BRIEF.md). The diagrams below retain the earlier experiments.

This document describes the implemented project as of **3 October 2026**.
The system verifies existing MCQs, generates and checks practice MCQs, and
compares its answering behaviour with vanilla RAG through a custom evaluation
harness. All three flows use the same Constitution and NCERT corpus.

The [standalone Mermaid source](diagrams/overall-system.mmd) can be opened in a
Mermaid editor for presentation or exported as SVG. The diagrams below explain
the overall architecture and the decisions inside each flow.

![Overall system architecture](diagrams/system-design.png)

Download the presentation image as [PNG](diagrams/system-design.png),
[SVG](diagrams/system-design.svg), or [PDF](diagrams/system-design.pdf).
Regenerate these exports with
`.venv/bin/python docs/diagrams/render_system_design.py`.

## 1. Overall architecture

Solid arrows show prepared data, evidence, results, or a requested workflow.
Double arrows show a query and its result. Dotted arrows show model calls and
revision feedback. These are module interactions; the detailed diagrams below
show execution order.

```mermaid
flowchart TB
    USER["Student / examiner"]
    CLI["Verification CLI"]

    subgraph KNOWLEDGE["1. Offline knowledge preparation"]
        PDFS["Constitution of India + NCERT Polity PDFs"]
        PREP["PyMuPDF extraction + cleaning<br/>Preserve document, source and page"]
        CHUNKING["1,000-character chunks<br/>150-character overlap"]
        CORPUS[("chunks.jsonl<br/>1,149 chunks")]
        PDFS --> PREP --> CHUNKING --> CORPUS
    end

    subgraph EVIDENCE["2. Shared evidence layer - local models"]
        INDEX["BGE-small embeddings<br/>In-memory NumPy index"]
        RETRIEVAL["Semantic search: top 20<br/>Cross-encoder rerank: top 5"]
        SCAN["Python full-corpus lexical scan<br/>Recognized quoted absence claims"]
        INDEX --> RETRIEVAL
    end
    CORPUS --> INDEX
    CORPUS --> SCAN

    subgraph APPLICATION["3. Tutor application"]
        TUTOR["Streamlit tutor<br/>Verify a question / Generate practice"]
        VERIFY["Question verifier<br/>Statements OR direct / best-answer MCQs<br/>Blind evidence review + Python source checks"]
        GENERATE["LangGraph practice workflow<br/>Generate, verify facts and key, audit quality<br/>Python ACCEPT / REVISE / REJECT"]
        BASELINE["Vanilla RAG comparison<br/>Retrieve, then LLM selects A-D"]
        OUTPUT["Results shown to the student<br/>Answer or abstention + reasons and sources<br/>Accepted practice MCQ or rejection"]
        TUTOR -->|Verify| VERIFY
        TUTOR -->|Practice| GENERATE
        VERIFY --> OUTPUT
        GENERATE --> OUTPUT
        GENERATE -. Revision feedback .-> GENERATE
    end
    USER --> TUTOR
    USER --> CLI
    CLI --> VERIFY
    VERIFY <-->|Queries / evidence| RETRIEVAL
    GENERATE <-->|Gate queries / evidence| RETRIEVAL
    BASELINE <-->|Full-question query / evidence| RETRIEVAL
    VERIFY <-->|Absence claim / lexical verdict| SCAN
    GENERATE <-->|Absence claim / lexical verdict| SCAN

    OPENAI["OpenAI API<br/>Runtime: gpt-4o-mini<br/>Earlier explanation judge: gpt-4o"]
    VERIFY -. Structured verification calls .-> OPENAI
    GENERATE -. Generation and gate calls .-> OPENAI
    BASELINE -. Answer call .-> OPENAI

    subgraph EVALUATION["4. Evaluation and report storage"]
        DATASET["Official UPSC PYQ benchmark<br/>75 questions: 33 development / 42 test<br/>Now used for regression after debugging"]
        HARNESS["Benchmark runner<br/>Verifier vs vanilla RAG<br/>Resumable checkpoints + run configuration"]
        SCORING["Python scoring<br/>Correct / wrong / abstained / errors<br/>Accuracy, coverage, precision and breakdowns"]
        DIAGNOSTICS["Earlier evaluation runners<br/>13-question regression, retrieval experiments,<br/>grounding, judge, stability and generation"]
        REPORTS[("data/evaluation/<br/>JSON results, traces and metrics")]
        EVALS["Evaluation Studio<br/>Tutor page + standalone evals_app.py<br/>Inspect reports / export CSV and JSON"]
        DATASET -->|Questions and options| HARNESS
        DATASET -->|Official keys for scoring| SCORING
        HARNESS -->|Run verifier| VERIFY
        HARNESS -->|Run baseline| BASELINE
        VERIFY -->|Predictions and traces| SCORING
        BASELINE -->|Predictions and traces| SCORING
        SCORING --> REPORTS
        DIAGNOSTICS --> REPORTS
        REPORTS -->|Read saved reports| EVALS
    end
    USER --> EVALS
    DIAGNOSTICS -. Judge and diagnostic model calls .-> OPENAI

    classDef interface fill:#eff6ff,stroke:#2563eb,color:#0f172a;
    classDef store fill:#f8fafc,stroke:#475569,color:#0f172a;
    classDef local fill:#ecfdf5,stroke:#059669,color:#0f172a;
    classDef model fill:#faf5ff,stroke:#9333ea,color:#0f172a;
    classDef evaluation fill:#fffbeb,stroke:#d97706,color:#0f172a;
    class USER,CLI,TUTOR,EVALS,OUTPUT interface;
    class CORPUS,REPORTS store;
    class PREP,CHUNKING,INDEX,RETRIEVAL,SCAN local;
    class OPENAI model;
    class DATASET,HARNESS,SCORING,DIAGNOSTICS evaluation;
```

Read this diagram in four parts:

1. **Prepare knowledge.** Extract PDF pages, clean their text and create chunks
   while retaining source and page metadata. Save the chunks in JSONL.
2. **Find evidence.** Embed the chunks with `BAAI/bge-small-en-v1.5`. A query
   retrieves 20 candidates by normalized embedding similarity. The
   `cross-encoder/ms-marco-MiniLM-L-6-v2` scores query–passage pairs and keeps
   five passages for that query.
3. **Check questions.** The verifier checks claims or answer options against
   the complete retrieved pages, including footnotes. Python validates exact
   source quotations and requires a second blind evidence review for an answer
   candidate. Python resolves the final answer or abstention. The
   practice workflow uses verification gates and revision feedback to decide
   whether to accept a generated question.
4. **Measure outcomes.** Evaluation runners record predictions, evidence and
   errors, score predictions against official keys, and save JSON reports.
   Evaluation Studio reads these reports and offers inspection and exports.

## 2. Existing-question verification

Numbered statements and direct MCQs have different evidence checks. The
question's official answer is withheld from both answering paths.

```mermaid
flowchart TB
    QUESTION["Question text + A-D options"] --> FORMAT{"Paired Statement-I/II<br/>or Assertion/Reason labels?"}
    FORMAT -->|Yes| UNSUPPORTED["ABSTAIN<br/>Separate statement and relationship checks unavailable"]
    FORMAT -->|No| ROUTE{"Question type?"}

    subgraph STATEMENTS["Numbered-statement path"]
        PARSE["Python parser<br/>Stem, items, closing and options"]
        BUILD["Python claim builder<br/>Bind each statement to its predicate<br/>Track source words and propositional form"]
        CLAIMSEARCH["For each claim: semantic top 20<br/>Cross-encoder rerank to top 5"]
        CLAIMGUARD{"Python claim metadata checks pass?<br/>Propositional form + source words"}
        GUARDFAIL["INSUFFICIENT<br/>Invalid or unbound claim"]
        PAGECONTEXT["Restore complete retrieved pages<br/>Include legal-status notes and exceptions"]
        ABSENCE{"FactVerifier recognizes a<br/>quoted lexical absence claim?"}
        SCAN["Python scans every loaded corpus chunk"]
        OCCURRENCE{"Quoted term found?"}
        CONTRADICTED["CONTRADICTED<br/>Return matching pages"]
        NOHIT["SUPPORTED for the loaded corpus<br/>Assumes scope coverage and faithful extraction"]
        FACTLLM["LLM: claim + complete source pages<br/>SUPPORTED / CONTRADICTED / INSUFFICIENT<br/>Reasoning + quoted evidence IDs"]
        QUOTECHECK["Python validates exact quotations<br/>and derives cited pages from source metadata"]
        COMMITTED{"Valid SUPPORTED or<br/>CONTRADICTED judgment?"}
        FACTREVIEW["Second LLM call: blind evidence review<br/>Same claim + context; first verdict withheld"]
        FACTAGREE["Python validates review quotations<br/>Require verdict agreement; else INSUFFICIENT"]
        AGGREGATE["Python aggregates claim verdicts<br/>into statement statuses"]
        MAP["Python maps statuses to coded options<br/>Strict policy is the default"]

        PARSE --> BUILD --> CLAIMSEARCH --> CLAIMGUARD
        CLAIMGUARD -->|No| GUARDFAIL --> AGGREGATE
        CLAIMGUARD -->|Yes| ABSENCE
        ABSENCE -->|Yes| SCAN --> OCCURRENCE
        OCCURRENCE -->|Yes| CONTRADICTED --> AGGREGATE
        OCCURRENCE -->|No| NOHIT --> AGGREGATE
        ABSENCE -->|No| PAGECONTEXT --> FACTLLM --> QUOTECHECK --> COMMITTED
        COMMITTED -->|No| AGGREGATE
        COMMITTED -->|Yes| FACTREVIEW --> FACTAGREE --> AGGREGATE
        AGGREGATE --> MAP
    end

    subgraph DIRECT["Direct / best-answer path"]
        QUERIES["Five searches<br/>Question + one query per option"]
        OPTIONSEARCH["Each search: top 20 to top 5<br/>Deduplicate into at most 25 passages"]
        OPTIONCONTEXT["Restore complete retrieved pages<br/>Check attached qualifications and footnotes"]
        OPTIONLLM["LLM compares answer fit for A-D<br/>SUPPORTED / RULED_OUT / INSUFFICIENT<br/>Reasoning + quoted evidence"]
        CITATIONS["Python validates evidence IDs<br/>and quotations against passage text"]
        RESOLVE["Python selects one SUPPORTED option<br/>only when all three alternatives are RULED_OUT"]
        DIRECTCANDIDATE{"Valid answer candidate?"}
        OPTIONREVIEW["Second LLM call: blind A-D evidence review<br/>Same question + context; first answer withheld"]
        DIRECTAGREE["Python applies quotation and option checks again<br/>Require the same answer; else abstain"]
        QUERIES --> OPTIONSEARCH --> OPTIONCONTEXT --> OPTIONLLM --> CITATIONS --> RESOLVE --> DIRECTCANDIDATE
        DIRECTCANDIDATE -->|Yes| OPTIONREVIEW --> DIRECTAGREE
    end

    ROUTE -->|Numbered statements| PARSE
    ROUTE -->|Direct / best answer| QUERIES
    MAP --> RESULT["Answer A-D + reasons and sources<br/>OR abstention with an explicit reason"]
    UNSUPPORTED --> RESULT
    DIRECTCANDIDATE -->|No| RESULT
    DIRECTAGREE --> RESULT
```

The current statement callers retrieve evidence before calling `FactVerifier`.
The shared claim guard rejects claims marked non-propositional or containing
unsupported introduced words before a model call. The verifier restores full
page context before a model call. Missing or invalid quotations downgrade a
committed model judgment to INSUFFICIENT. A valid committed judgment needs a
second blind review and agreement on its verdict. Direct answer candidates need
a second valid option analysis selecting the same answer. Both calls use the
same configured model; agreement does not establish statistical independence or
guarantee correctness.

The direct entry points recognize unsupported paired labels before option
retrieval or a model call. They return an explicit abstention for these formats.
Generic option comparisons do not stand in for independent statement and
explanatory-relationship judgments.
For a recognized quoted absence claim, the verifier then uses the full-corpus
lexical scan to return a verdict. Other claims use the retrieved passages in an
LLM verification. This scan addresses literal occurrence of a quoted term; broader
conceptual claims follow ordinary evidence verification.

For example, “there is no mention of the word ‘emergency’ in the loaded corpus”
can be contradicted by a literal occurrence in that corpus. Finding “emergency” in a passage
does not by itself adjudicate a claim such as “the President can declare an
emergency without approval.” That claim requires evidence about the asserted
power and conditions.

The direct path checks whether an option answers the exact question. An option
can describe a true secondary function and still be ruled out for a “chief
purpose” question. Quotation matching verifies where the quoted words came
from; whether those words justify the assessment remains a model judgment.

## 3. Practice generation and verification

The LangGraph workflow carries a shared `MCQVerificationState` through six
nodes. Pydantic models define the generated MCQ and the gate results.

```mermaid
flowchart TB
    TOPIC["Topic + difficulty + question format"] --> GENERATE["1. Generate MCQ<br/>LLM proposes question, options, key and explanation"]
    GENERATE --> EXTRACT["2. Extract claims<br/>Deterministic construction first<br/>LLM fallback when needed"]
    EXTRACT --> FACTS["3. Verify claims<br/>Fresh retrieval for each claim + FactVerifier"]
    FACTS --> KEY["4. Verify answer key<br/>Retrieve evidence for the question<br/>LLM assesses options with declared key withheld<br/>Python compares assessment with the generated key"]
    KEY --> QUALITY["5. Audit quality<br/>Ambiguity, wording, answer uniqueness,<br/>distractors and topic relevance"]
    QUALITY --> DECIDE{"6. Python decision<br/>All three gates pass?"}
    DECIDE -->|Yes| ACCEPT["ACCEPT<br/>Show the accepted practice MCQ"]
    DECIDE -->|No| BUDGET{"Revisions available?"}
    BUDGET -->|Yes| FEEDBACK["REVISE<br/>Feed failure reasons into the next prompt"]
    FEEDBACK --> GENERATE
    BUDGET -->|No| REJECT["REJECT<br/>Show failed gates and reasons"]
```

The default budget is **one initial generation plus two revisions**, for at most
three attempts. The initial generator uses the topic and prompt; retrieval
happens in the verification gates after generation. `ACCEPT` means the implemented
gates passed for that candidate.

## 4. Evaluation framework and dashboard

The expanded benchmark runner compares the source verifier with vanilla RAG on
the frozen development and test splits. The two arms share a corpus and
retriever instance within a benchmark run. Each arm constructs its own queries.

```mermaid
flowchart TB
    OFFICIAL["Official UPSC papers + booklet-matched keys<br/>Source PDFs and hashes"]
    OFFICIAL --> DATASET["Frozen polity_v1 benchmark<br/>75 questions: development 33 / test 42"]
    DATASET --> VALIDATE["Validate sources, labels and split IDs<br/>Record dataset, corpus, code and model configuration"]
    VALIDATE --> INPUT["Answering boundary<br/>Question text + options only"]
    INPUT --> VERIFIED["Source verifier<br/>Statement / direct routing<br/>Answer or abstain"]
    INPUT --> VANILLA["Vanilla RAG<br/>Whole-question retrieval<br/>LLM chooses A-D"]
    VERIFIED --> CHECKPOINT["Checkpoint after each question / system<br/>Predictions, evidence, reasoning and service errors"]
    VANILLA --> CHECKPOINT
    CHECKPOINT --> SCORE["Python scoring<br/>Correct, wrong, abstained and errors<br/>Accuracy, coverage, precision and practice marks<br/>Breakdowns by format, topic and year"]
    DATASET -->|Official keys| SCORE
    SCORE --> REPORTS[("development_results.json<br/>test_results.json")]
    LEGACY["Earlier regression and diagnostic reports<br/>13-question answer quality and citation presence<br/>LLM explanation judge and verdict stability<br/>Retrieval and generation experiments"] --> DASHBOARD
    REPORTS --> DASHBOARD["Evaluation Studio<br/>Validate report scope, inspect traces,<br/>refresh results, download CSV / JSON"]
```

The 75-question reports contain answer-quality metrics and detailed traces.
The test split has now informed debugging of an observed date/event error.
Its fixed IDs remain unchanged; current runs on both splits are regression
validation. An independent accuracy estimate requires fresh untouched data.
The dashboard's citation-presence, explanation-judge and verdict-stability views
use the earlier 13-question regression reports. Generation and retrieval
experiments have their own datasets and saved outputs. These measurements keep
their recorded scope.

Citation presence is a grounding proxy. The earlier explanation judge scores
correctness, faithfulness, reasoning and honesty with `gpt-4o`; those are model
ratings. API or service failures are recorded separately from deliberate
abstentions. The benchmark runner supports checkpoint resume and explicit
retry of recorded errors.

## 5. Deployment and storage

This is a local Python application with imported modules. Each Streamlit process
has its own memory and resource cache; the benchmark runner is another process
when launched from the terminal.

```mermaid
flowchart LR
    BROWSER["Browser"] --> TUTOR["Streamlit tutor<br/>app.py - usual port 8501"]
    BROWSER --> EVALS["Standalone Evaluation Studio<br/>evals_app.py - configured port 8502"]
    TUTOR --> MODULES["Python modules<br/>Verification + retrieval + LangGraph"]
    MODULES --> MEMORY["Process memory<br/>Model weights + chunk embeddings"]
    MODULES --> OPENAI["OpenAI API<br/>Live LLM calls"]
    TERMINAL["Terminal"] --> RUNNER["Evaluation runners / verification CLI"]
    RUNNER --> MODULES
    FILES[("Local files<br/>Source PDFs, corpus JSONL,<br/>benchmark JSON and evaluation reports")]
    MODULES -->|Load corpus| FILES
    RUNNER -->|Read inputs / save results| FILES
    EVALS -->|Read saved reports| FILES
    TUTOR -->|Embedded Evaluation page| FILES
```

The retrieval index is a normalized NumPy embedding matrix created when the
retriever initializes. Streamlit caches the corpus and model resources within
its process. Persistent project data is stored in local files. Evaluation
Studio refreshes report files through its shared dashboard modules.

## 6. Components and code

| Component | Implementation | Responsibility |
|---|---|---|
| Tutor and CLI | [app.py](../app.py), [verify_cli.py](../src/verify_cli.py) | Question input, verification results and practice generation |
| Evaluation UI | [evals_app.py](../evals_app.py), [evaluations.py](../src/ui/evaluations.py), [evaluation_data.py](../src/ui/evaluation_data.py) | Shared report views, validation, refresh and exports |
| Ingestion | [prepare_documents.py](../src/ingestion/prepare_documents.py), [chunk_documents.py](../src/ingestion/chunk_documents.py), [save_chunks_jsonl.py](../src/ingestion/save_chunks_jsonl.py) | PDF text, metadata, chunking and corpus persistence |
| Dense retrieval | [semantic_retriever.py](../src/retrieval/semantic_retriever.py) | Local BGE embeddings and NumPy similarity ranking |
| Reranking | [semantic_reranker.py](../src/retrieval/semantic_reranker.py), [retrieval_config.py](../src/retrieval/retrieval_config.py), [claim_retriever.py](../src/verification/claim_retriever.py) | Cross-encoder ranking and shared 20-to-5 depth settings |
| Claim construction | [question_parser.py](../src/verification/question_parser.py), [claim_builder.py](../src/verification/claim_builder.py), [claim_extractor.py](../src/verification/claim_extractor.py) | Parse numbered items, bind context and construct claims; generation supports LLM fallback |
| Fact verification | [fact_verifier.py](../src/verification/fact_verifier.py), [negative_claim_checker.py](../src/verification/negative_claim_checker.py) | Claim guards, quoted judgments and blind review, or recognized lexical absence scans |
| Checked evidence | [evidence_context.py](../src/verification/evidence_context.py) | Restore page-level context after ranking, preserve source identity and trace context chunks |
| Answer decisions | [answer_mapping.py](../src/evaluation/answer_mapping.py), [direct_mcq_verifier.py](../src/verification/direct_mcq_verifier.py) | Strict statement mapping or direct-option resolution with quotation checks and blind review |
| Practice workflow | [mcq_generator.py](../src/generation/mcq_generator.py), [graph.py](../src/orchestration/graph.py), [nodes.py](../src/orchestration/nodes.py), [state.py](../src/orchestration/state.py) | LangGraph generation, three gates, feedback and retry budget |
| Generation gates | [answer_key_verifier.py](../src/verification/answer_key_verifier.py), [mcq_quality_auditor.py](../src/verification/mcq_quality_auditor.py) | Independently analyze the generated answer key and audit question quality |
| Comparison baseline | [vanilla_rag.py](../src/rag/vanilla_rag.py) | Retrieval followed by one LLM answer selection |
| Expanded benchmark | [benchmark.py](../src/evaluation/benchmark.py), [evaluate_benchmark.py](../src/evaluation/evaluate_benchmark.py) | Frozen input validation, gold-answer isolation, model runs, checkpoints and scoring |
| Earlier evaluation chain | [run_all.py](../src/evaluation/run_all.py) | Dependency-ordered regression stages, freshness and retrieval-depth checks |

See [the benchmark guide](BENCHMARK.md) for recorded results, [the Evals UI
guide](EVALS_UI.md) for dashboard usage, and [the detailed system design](SYSTEM_DESIGN.md)
for the earlier module contracts and research record.
See [the verifier reliability note](VERIFIER_RELIABILITY.md) for the development
error analysis and the new quotation and page-context checks.
