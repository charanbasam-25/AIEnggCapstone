from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation.answer_mapping import map_answer
from src.evaluation.benchmark import (
    BENCHMARK_PATH,
    LOCK_PATH,
    content_fingerprint,
    inventory,
    load_benchmark,
    normalise_numbered_question,
    question_input,
    score_records,
    validate_dataset,
)
from src.evaluation.build_benchmark import build_dataset
from src.evaluation import evaluate_benchmark as runner


@pytest.fixture
def dataset():
    return load_benchmark()


def test_frozen_dataset_and_official_sources_are_reproducible(dataset):
    validate_dataset(dataset, check_sources=True)
    rebuilt = (json.dumps(build_dataset(), indent=2, ensure_ascii=False) + "\n").encode()
    lock = json.loads(LOCK_PATH.read_text())
    assert rebuilt == BENCHMARK_PATH.read_bytes()
    assert hashlib.sha256(rebuilt).hexdigest() == lock["dataset_sha256"]
    assert inventory(dataset)["splits"] == {"development": 33, "test": 42}
    assert inventory(dataset)["question_types"] == {
        "direct_mcq": 29, "statement_mcq": 41, "best_answer_mcq": 5,
    }
    assert len(dataset["questions"]) == 75


def test_old_questions_remain_regression_only(dataset):
    old = json.loads(Path("data/evaluation/pyq_2025_polity.json").read_text())["questions"]
    assert len(old) == 13
    old_content = {content_fingerprint(q) for q in old}
    assert not old_content & {content_fingerprint(q) for q in dataset["questions"]}
    assert not any(q["year"] == 2025 for q in dataset["questions"])


@pytest.mark.parametrize("year,number,answer", [
    (2019, 56, "B"), (2020, 7, "D"), (2021, 79, "C"),
    (2021, 90, "D"), (2022, 73, "A"), (2023, 33, "C"),
])
def test_reviewed_booklet_key_examples(dataset, year, number, answer):
    question = next(q for q in dataset["questions"] if q["year"] == year and q["q_number"] == number)
    assert question["booklet_series"] == "A"
    assert question["official_answer"] == answer


@pytest.mark.parametrize("change,message", [
    (lambda q: q.update(official_answer="A"), "Official-key mismatch"),
    (lambda q: q.update(booklet_series="B"), "series mismatch"),
    (lambda q: q["options"].pop("D"), "four nonempty options"),
    (lambda q: q.update(source_pdf_pages=[999]), "PDF page"),
    (lambda q: q.update(question_text="Changed wording"), "text transformation"),
    (lambda q: q.update(official_answer="X"), "dropped answer"),
])
def test_validation_rejects_label_and_source_damage(dataset, change, message):
    changed = deepcopy(dataset)
    change(changed["questions"][0])
    with pytest.raises(ValueError, match=message):
        validate_dataset(changed)


def test_duplicate_and_development_leakage_checks(dataset):
    changed = deepcopy(dataset)
    changed["questions"].append(deepcopy(changed["questions"][0]))
    with pytest.raises(ValueError, match="Duplicate question ID"):
        validate_dataset(changed)
    changed["questions"][-1]["id"] = "different-id"
    changed["questions"][-1]["split"] = "test"
    with pytest.raises(ValueError, match="Duplicate question content"):
        validate_dataset(changed)
    changed = deepcopy(dataset)
    example = next(q for q in changed["questions"] if q["id"] == "upsc-2023-a-q033")
    assert example["split"] == "development"
    example["split"] = "test"
    with pytest.raises(ValueError, match="development example"):
        validate_dataset(changed)
    assert not any(q["q_number"] == 80 and q["year"] == 2021 for q in dataset["questions"])
    assert not any(q["q_number"] == 34 and q["year"] == 2023 for q in dataset["questions"])


def test_normalisation_changes_labels_without_changing_facts():
    text = "Consider the following:\n1. Article 32 protects rights.\n2. A term lasts 5 years.\nWhich are correct?"
    options = {"A": "1 only", "B": "2 only", "C": "Both 1 and 2", "D": "Neither 1 nor 2"}
    normal, mapped, operations = normalise_numbered_question(text, options)
    assert "I. Article 32" in normal
    assert "II. A term lasts 5 years" in normal
    assert mapped["C"] == "Both I and II"
    assert operations == ["arabic_item_ids_to_roman"]
    assert options["C"] == "Both 1 and 2"  # No input mutation.
    assert normalise_numbered_question("Article 1 applies. Article 2 applies.", options)[2] == []
    assert normalise_numbered_question("1. One\n3. Three", options)[2] == []


def test_five_item_option_mapping_does_not_drop_the_fifth_item():
    statuses = {label: "CONTRADICTED" for label in ("I", "II", "III", "IV")}
    statuses["V"] = "SUPPORTED"
    answer, _ = map_answer(statuses, {"A": "I only", "B": "II only", "C": "V only", "D": "None"}, "Which statements are correct?")
    assert answer == "C"


def test_scoring_separates_abstentions_service_errors_and_pending(dataset):
    questions = dataset["questions"][:5]
    wrong = next(letter for letter in "ABCD" if letter != questions[1]["official_answer"])
    rows = [
        {"question_id": questions[0]["id"], "status": "ANSWERED", "predicted_answer": questions[0]["official_answer"]},
        {"question_id": questions[1]["id"], "status": "ANSWERED", "predicted_answer": wrong},
        {"question_id": questions[2]["id"], "status": "ABSTAINED", "predicted_answer": None},
        {"question_id": questions[3]["id"], "status": "ERROR", "predicted_answer": None},
    ]
    metrics = score_records(questions, rows)["overall"]
    assert (metrics["correct"], metrics["wrong"], metrics["abstained"], metrics["errors"], metrics["pending"]) == (1, 1, 1, 1, 1)
    assert metrics["coverage"] == .4
    assert metrics["precision_when_answered"] == .5
    assert metrics["accuracy_overall"] == .2
    assert metrics["error_rate_overall"] == .2
    assert metrics["upsc_marks"] == pytest.approx(4 / 3)
    assert score_records(questions, [rows[2]])["overall"]["precision_when_answered"] is None
    with pytest.raises(ValueError, match="duplicate"):
        score_records(questions, rows + [rows[0]])
    with pytest.raises(ValueError, match="disagree"):
        score_records(questions, [{**rows[2], "predicted_answer": "A"}])


def test_answering_input_excludes_the_key_and_is_independent(dataset):
    question = dataset["questions"][0]
    given = question_input(question)
    assert set(given) == {"question_text", "options"}
    given["options"]["A"] = "Changed option"
    assert given["options"] != question["options"]


@pytest.mark.parametrize("mode", ["--check", "--dry-run"])
def test_offline_modes_never_initialise_answering_services(monkeypatch, capsys, mode):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline command loaded models or an API client")
    monkeypatch.setattr(runner, "make_services", forbidden)
    runner.main([mode, "--split", "test"])
    output = capsys.readouterr().out
    if mode == "--check":
        assert "PASS" in output
    else:
        assert "no API calls made" in output


def test_test_split_cannot_be_selectively_limited():
    with pytest.raises(SystemExit):
        runner.main(["--dry-run", "--split", "test", "--limit", "3"])


def test_paired_comparison_uses_exactly_the_questions_both_answered(dataset):
    questions = dataset["questions"][:3]
    def row(question, prediction):
        return {
            "question_id": question["id"], "predicted_answer": prediction,
            "status": "ANSWERED" if prediction else "ABSTAINED",
        }
    correct = [q["official_answer"] for q in questions]
    wrong = next(letter for letter in "ABCD" if letter != correct[1])
    results = {
        "verified": [row(questions[0], correct[0]), row(questions[1], correct[1]), row(questions[2], None)],
        "vanilla": [row(questions[0], correct[0]), row(questions[1], wrong), row(questions[2], correct[2])],
    }
    comparison = runner.paired_summary(questions, results)
    assert comparison["both_answered"] == 2
    assert comparison["both_correct"] == 1
    assert comparison["verified_only_correct"] == 1
    assert comparison["question_ids"] == [q["id"] for q in questions[:2]]


def test_checkpoint_resume_and_failures_do_not_leak_labels(dataset, tmp_path):
    questions = dataset["questions"][:2]
    inputs = []
    def verified(given):
        assert set(given) == {"question_text", "options"}
        inputs.append(given)
        if len(inputs) == 1:
            return {"status": "ABSTAINED", "predicted_answer": None}
        raise RuntimeError("PRIVATE_API_ERROR_DATA")
    def vanilla(given):
        assert set(given) == {"question_text", "options"}
        return {"status": "ANSWERED", "predicted_answer": "A"}
    config = {"systems": ["verified", "vanilla"], "dataset_sha256": "fixed"}
    path = tmp_path / "results.json"
    report = runner.run_evaluation(questions, config, {"verified": verified, "vanilla": vanilla}, path)
    assert report["complete"]
    assert report["metrics"]["verified"]["overall"]["errors"] == 1
    assert report["metrics"]["verified"]["overall"]["abstained"] == 1
    assert "PRIVATE_API_ERROR_DATA" not in path.read_text()
    # Fully completed rows, including errors, stay cached unless retries were requested.
    assert runner.run_evaluation(questions, config, {}, path)["results"] == report["results"]
    assert len(inputs) == 2
    with pytest.raises(ValueError, match="settings changed"):
        runner.run_evaluation(questions, {**config, "dataset_sha256": "changed"}, {}, path)
    retried = runner.run_evaluation(
        questions, config,
        {"verified": lambda given: {"status": "ANSWERED", "predicted_answer": "C"}},
        path, retry_errors=True,
    )
    assert retried["metrics"]["verified"]["overall"]["errors"] == 0
    assert retried["metrics"]["verified"]["overall"]["abstained"] == 1
    assert retried["results"]["vanilla"] == report["results"]["vanilla"]


def test_shared_verifier_routes_statement_and_direct_inputs(dataset):
    from src.verification.fact_verifier import FactEvidenceCitation, FactVerificationResult
    retrieved, verified = [], []
    class Retriever:
        def retrieve(self, text, top_k):
            retrieved.append((text, top_k))
            return [{"source": "test", "page": 1, "text": "Source evidence"}]
    class Facts:
        def verify(self, claim, evidence):
            return FactVerificationResult(
                verdict="SUPPORTED", reasoning="Synthetic test verdict", supporting_pages=[1],
                independently_reviewed=True, checked_evidence=evidence,
                citations=[FactEvidenceCitation(evidence_id=1, quote="Source evidence")],
            )
    class Direct:
        def verify(self, text, options, evidence):
            verified.append((text, options))
            return SimpleNamespace(model_dump=lambda: {"status": "ANSWERED", "predicted_answer": "A"})
    statement = next(q for q in dataset["questions"] if q["id"] == "upsc-2020-a-q013")
    result = runner.verify_question(question_input(statement), Retriever(), Facts(), Direct())
    assert result["predicted_answer"] == "C"
    assert len(retrieved) == 2 and not verified
    direct = dataset["questions"][0]
    result = runner.verify_question(question_input(direct), Retriever(), Facts(), Direct())
    assert result["predicted_answer"] == "A"
    assert verified[0][0] == direct["question_text"]


@pytest.mark.parametrize("reviewed,quote", [(False, "Source evidence"), (True, "Absent quotation")])
def test_benchmark_blocks_unreviewed_or_unbound_model_claims(dataset, reviewed, quote):
    from src.verification.fact_verifier import FactEvidenceCitation, FactVerificationResult
    class Retriever:
        def retrieve(self, text, top_k):
            return [{"source": "synthetic", "page": 1, "text": "Source evidence"}]
    class Facts:
        def verify(self, claim, evidence):
            return FactVerificationResult(
                verdict="SUPPORTED", reasoning="Synthetic proposed verdict", supporting_pages=[1],
                independently_reviewed=reviewed, checked_evidence=evidence,
                citations=[FactEvidenceCitation(evidence_id=1, quote=quote)],
            )
    statement = next(q for q in dataset["questions"] if q["id"] == "upsc-2020-a-q013")
    result = runner.verify_question(question_input(statement), Retriever(), Facts(), None)
    assert result["status"] == "ABSTAINED"
    assert result["predicted_answer"] is None
    assert all(row["verification"]["validation_issues"] for row in result["claims"])


def test_benchmark_article_context_and_real_reference_reviews_survive_a_search_miss():
    from src.verification.article_context import ArticleEvidenceContext
    from src.verification.fact_verifier import FactVerifier, ReferencedFactEvidenceAssessment
    from src.verification.source_quotes import QuoteSelection
    # Invented rules are deliberately labelled synthetic; this tests the
    # workflow contract, not constitutional facts or live model accuracy.
    def page(number, text):
        return {"id": number, "source": "constitution", "document": "Synthetic test constitution",
                "page": number, "chunk_index": 0, "text": text}
    chunks = [
        page(1, "14. Earlier rule.—A different provision.\n"
                "15. Membership.—The committee has exactly seven members."),
        page(2, "A continuing qualification. The committee shall include one elected member."),
        page(3, "16. Next rule.—Another provision."),
        page(99, "Unrelated passage from semantic search."),
    ]
    requests, searches = [], []
    def parse(**kwargs):
        requests.append(kwargs)
        assert kwargs["text_format"] is ReferencedFactEvidenceAssessment
        assert "GOLD_KEY_SENTINEL" not in kwargs["input"]
        catalog = json.loads(kwargs["input"].split("SOURCE EVIDENCE:", 1)[1])
        assert any("one elected member" in row["quote"] for row in catalog)
        quote = next(row for row in catalog if "exactly seven members" in row["quote"])
        claim = kwargs["input"].split("CLAIM:", 1)[1].split("SOURCE EVIDENCE:", 1)[0]
        return SimpleNamespace(output_parsed=ReferencedFactEvidenceAssessment(
            verdict="CONTRADICTED" if "eleven" in claim else "SUPPORTED",
            reasoning="Synthetic source comparison.", citations=[QuoteSelection(quote_id=quote["quote_id"])],
        ))
    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    class Retriever:
        def retrieve(self, text, top_k):
            searches.append(text)
            return [chunks[-1]]
    given = {
        "question_text": "Consider the following statements:\n"
                         "I. Under Article 15, the committee has exactly seven members.\n"
                         "II. Under Article 15, the committee has exactly eleven members.\n"
                         "Which of the statements given above is/are correct?",
        "options": {"A": "I only", "B": "II only", "C": "Both I and II", "D": "Neither I nor II"},
        "official_answer": "GOLD_KEY_SENTINEL",
    }
    result = runner.verify_question(
        question_input(given), Retriever(), FactVerifier(chunks, client=client, reference_quotes=True), None,
        article_context=ArticleEvidenceContext(chunks),
    )
    assert result["status"] == "ANSWERED"
    assert result["predicted_answer"] == "A"
    assert len(searches) == 2 and len(requests) == 4
    for row in result["claims"]:
        checked = row["verification"]
        assert checked["independently_reviewed"]
        assert checked["supporting_pages"] == [1]
        assert [item["page"] for item in checked["checked_evidence"]] == [99, 1, 2, 3]


def test_benchmark_factory_enables_reference_binding_for_both_verifiers(monkeypatch):
    import src.orchestration.telemetry as telemetry
    import src.retrieval.semantic_reranker as retrieval
    import src.verification.fact_verifier as facts
    import src.verification.direct_mcq_verifier as direct
    seen, client = {}, object()
    monkeypatch.setattr(retrieval, "load_chunks", lambda path: [])
    monkeypatch.setattr(retrieval, "SemanticReranker", lambda *args, **kwargs: object())
    monkeypatch.setattr(telemetry, "create_model_client", lambda: client)
    def factory(label):
        def create(*args, **kwargs):
            seen[label] = kwargs
            return object()
        return create
    monkeypatch.setattr(facts, "FactVerifier", factory("facts"))
    monkeypatch.setattr(direct, "DirectMCQVerifier", factory("direct"))
    assert set(runner.make_services(["verified"])) == {"verified"}
    assert all(settings["reference_quotes"] is True and settings["client"] is client
               for settings in seen.values())


def test_benchmark_config_records_the_updated_protocol(dataset):
    questions = [q for q in dataset["questions"] if q["split"] == "development"][:5]
    config = runner.run_config(dataset, questions, "development", ["verified"])
    assert config["verification_pipeline"] == runner.VERIFICATION_PIPELINE
    assert config["request_timeout_seconds"] == 45
    assert config["sdk_transport_retries"] == 1
    assert not config["is_full_split"]


def test_benchmark_ui_shows_inventory_and_saved_splits_under_how_it_works():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    assert not app.exception, [item.message for item in app.exception]
    assert not app.metric
    app.radio(key="workspace").set_value("How it works").run()
    app.radio(key="about_section").set_value("Evals").run()
    assert not app.exception, [item.message for item in app.exception]
    inventory = {metric.label: metric.value for metric in app.metric}
    assert inventory["Benchmark questions"] == "75"
    assert inventory["Development"] == "33"
    assert inventory["Test"] == "42"
    assert app.selectbox(key="benchmark_split").options == ["Development", "Test"]
    app.selectbox(key="benchmark_split").set_value("test").run()
    assert not app.exception, [item.message for item in app.exception]
    assert "Verify a question" not in app.radio(key="workspace").options
