"""Prepare one practice question using the same publication gates as the UI."""

import argparse

from src.practice.catalog import TOPICS
from src.practice.models import PracticeRequest
from src.practice.service import prepare_practice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", choices=tuple(TOPICS), default="Fundamental Rights")
    parser.add_argument("--format", choices=("simple", "statements"), default="simple")
    parser.add_argument("--max-retries", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--fresh", action="store_true", help="Generate rather than reuse a checked item.")
    args = parser.parse_args()
    result = prepare_practice(PracticeRequest(
        topic=args.topic, count=1, question_format=args.format,
        max_retries=args.max_retries, reuse_checked=not args.fresh,
    ))
    report = result.report
    print(f"Prepared {report['accepted']}/{report['requested']} questions; "
          f"{report['api_calls']} model calls; {report['elapsed_seconds']:.1f}s.")
    if not result.questions:
        print("No question passed all the evidence checks.")
        if report["error_type"]:
            print(f"Run error: {report['error_type']}")
        return report
    for question in result.questions:
        mcq = question.mcq
        print(f"\n{mcq.question}")
        for letter in "ABCD":
            print(f"{letter}. {getattr(mcq, f'option_{letter.lower()}')}")
        print(f"\nReviewed answer: {mcq.correct_answer}\n{question.notes.summary.text}")
    print("\nAutomated evidence review; expert review is still needed before student release.")
    return report


if __name__ == "__main__":
    main()
