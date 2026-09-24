from src.orchestration.graph import build_graph


def main():
    workflow = build_graph()

    initial_state = {
        "topic": "Fundamental Rights",
        "difficulty": "medium",
        "retry_count": 0,
        "max_retries": 2,
    }

    result = workflow.invoke(initial_state)

    print("\n=== FINAL MCQ ===\n")

    mcq = result["mcq"]

    print(mcq.question)
    print(f"A. {mcq.option_a}")
    print(f"B. {mcq.option_b}")
    print(f"C. {mcq.option_c}")
    print(f"D. {mcq.option_d}")

    print(f"\nCorrect Answer: {mcq.correct_answer}")

    print("\n=== FACT VERIFICATION ===\n")

    for claim in result["claims"]:
        verification = result["fact_verifications"][claim.claim_id]

        print(f"\n{claim.claim_id}")
        print(f"Claim: {claim.claim}")
        print(f"Verdict: {verification.verdict}")
        print(f"Reasoning: {verification.reasoning}")
        print(
            f"Supporting pages: "
            f"{verification.supporting_pages}"
        )

    print("\n=== ANSWER-KEY VERIFICATION ===\n")

    answer = result["answer_verification"]

    print(f"Declared answer: {answer.declared_answer}")
    print(f"Supported options: {answer.supported_options}")
    print(f"Exactly one correct: {answer.exactly_one_correct}")
    print(f"Verdict: {answer.verdict}")
    print(f"Reasoning: {answer.reasoning}")
    print(f"Supporting pages: {answer.supporting_pages}")

    print("\n=== QUALITY AUDIT ===\n")

    quality = result["quality_audit"]

    print(f"Unambiguous: {quality.unambiguous}")
    print(f"Single best answer: {quality.single_best_answer}")
    print(
        f"Plausible distractors: "
        f"{quality.plausible_distractors}"
    )
    print(
        f"Appropriate wording: "
        f"{quality.appropriate_wording}"
    )
    print(f"Topic relevant: {quality.topic_relevant}")
    print(f"Overall quality: {quality.overall_quality}")

    if quality.issues:
        print("\nQuality issues:")

        for issue in quality.issues:
            print(f"- {issue}")

    print("\n=== FINAL DECISION ===")

    print(f"Decision: {result['decision']}")
    print(f"Attempts: {result['retry_count']}")

    if result["failure_reasons"]:
        print("\nFailure reasons:")

        for reason in result["failure_reasons"]:
            print(f"- {reason}")


if __name__ == "__main__":
    main()