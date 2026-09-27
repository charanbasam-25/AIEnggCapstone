"""
Audit of the gold answer labels in the 13-question evaluation set.

Why this exists
---------------

Every accuracy number in this project is measured against
`official_answer` in verified_pyq_results.json. That field was taken on
trust. It should not have been: the ablation run surfaced a question
where five of six arms agreed on an answer that the label called wrong,
and the corpus turned out to side with the arms, not the label.

A wrong label is worse than a wrong prediction, because it inverts the
gradient. A fix that makes the system more correct is scored as a
regression, and the fix gets reverted. That is exactly what nearly
happened here: the label error made a false SUPPORTED look like the
project's single best result.

Scope, and its limit
--------------------

Only labels that some arm contradicted could be caught this way, so only
those were audited in depth. The three are recorded below with the corpus
text that settles each. Six further labels are corroborated, in the weak
sense that the pipeline derived the same answer independently from the
corpus. Four questions are abstained by every arm and so are unaudited;
they are listed as such rather than assumed correct.

This module changes no metric by itself. `corrected_label` is applied
only where `disputed` is True, and the pipeline reports both scorings so
that the effect of the correction is visible rather than absorbed.
"""


AUDIT_CORROBORATED = "corroborated"
AUDIT_CONFIRMED = "confirmed_correct"
AUDIT_DISPUTED = "disputed"
AUDIT_UNAUDITED = "unaudited"


LABEL_AUDIT = {
    54: {
        "status": AUDIT_CONFIRMED,
        "stored_label": "A",
        "corrected_label": "A",
        "disputed": False,
        "system_answer": "D",
        "finding": (
            "The label is correct and the system is wrong. Article 368(2) "
            "makes State ratification conditional on a closed list of "
            "subjects. Items I (a List in the Seventh Schedule) and II "
            "(extent of the executive power of a State, article 162) are "
            "on that list; item III, conditions of the Governor's office, "
            "is not. The verifier returned SUPPORTED for all three "
            "because the evidence states the ratification rule and the "
            "verifier confirmed that the rule exists rather than that "
            "this subject falls under it."
        ),
        "failure_mode": "closed_list_membership",
    },
    58: {
        "status": AUDIT_DISPUTED,
        "stored_label": "A",
        "corrected_label": "D",
        "disputed": True,
        "system_answer": "D",
        "finding": (
            "The stored label 'I only' cannot be right, because both "
            "statements are false against this corpus.\n\n"
            "Statement I says that a Tenth Schedule disqualification of a "
            "Member of the House of the People is decided by the "
            "President, in accordance with the opinion of the Council of "
            "Union Ministers. It is false twice over. Tenth Schedule "
            "paragraph 6(1) (p. 378) refers the question 'for the "
            "decision of the Chairman or, as the case may be, the Speaker "
            "of such House and his decision shall be final'. Article 103 "
            "does give the decision to the President, but only for the "
            "disqualifications 'mentioned in clause (1) of article 102' "
            "(p. 77), whereas Tenth Schedule disqualification is clause "
            "(2); and article 103(2) (p. 78) directs the President to "
            "obtain the opinion of the Election Commission, not of the "
            "Council of Ministers.\n\n"
            "Statement II says the Constitution does not mention "
            "'political party'. An exhaustive case-insensitive scan of "
            "all 1149 corpus chunks finds the phrase 12 times, on pages "
            "65, 104, 251, 376, 377, 378 and 379, including the Tenth "
            "Schedule and articles 75(1B) and 164(1B). Two of those "
            "pages, 65 and 104, never appeared in any top-k retrieval "
            "for this claim, which is why the scan is exhaustive rather "
            "than ranked.\n\n"
            "Both false, so the answer is D, 'Neither I nor II'."
        ),
        "failure_mode": "wrong_gold_label",
    },
    65: {
        "status": AUDIT_CONFIRMED,
        "stored_label": "D",
        "corrected_label": "D",
        "disputed": False,
        "system_answer": "B",
        "finding": (
            "The label is correct and the system is half wrong. Item I is "
            "correctly CONTRADICTED once bound: the Fifth Schedule "
            "provides that the executive power of the State extends to "
            "Scheduled Areas, so the State does not lose it. Item II is a "
            "false SUPPORTED. The evidence establishes that the Governor "
            "reports to the President and that the Union may give "
            "directions on the administration of Scheduled Areas; the "
            "claim asserts that the Union can take over the total "
            "administration. The verifier accepted evidence for a weaker "
            "proposition as establishing a stronger one."
        ),
        "failure_mode": "strength_inflation",
    },
}

# Labels the pipeline reproduced independently from the corpus. That is
# weak positive evidence, not verification: a shared misreading would
# agree just as neatly. They are recorded so the audit's coverage is
# explicit rather than implied.
CORROBORATED = (56, 57, 59, 61, 62, 63)

# Abstained by every arm, so nothing was ever asserted against them and
# this method could not test them either way.
UNAUDITED = (55, 60, 64, 66)


def corrected_label(q_number: int, stored_label: str) -> str:
    """Return the audited label, falling back to the stored one."""

    entry = LABEL_AUDIT.get(q_number)

    if entry is not None and entry["disputed"]:
        return entry["corrected_label"]

    return stored_label


def audit_status(q_number: int) -> str:
    """Classify how much is known about one question's label."""

    entry = LABEL_AUDIT.get(q_number)

    if entry is not None:
        return entry["status"]

    if q_number in CORROBORATED:
        return AUDIT_CORROBORATED

    return AUDIT_UNAUDITED


def disputed_questions() -> "list[int]":
    return sorted(
        q for q, entry in LABEL_AUDIT.items() if entry["disputed"]
    )


def self_check() -> None:
    """Guard the audit's internal consistency."""

    covered = set(LABEL_AUDIT) | set(CORROBORATED) | set(UNAUDITED)

    assert len(covered) == 13, (
        f"audit must account for all 13 questions, has {len(covered)}"
    )

    # No question may be classified twice; overlapping groups would make
    # the coverage claim above meaningless.
    assert not set(LABEL_AUDIT) & set(CORROBORATED)
    assert not set(LABEL_AUDIT) & set(UNAUDITED)
    assert not set(CORROBORATED) & set(UNAUDITED)

    for q_number, entry in LABEL_AUDIT.items():

        # A disputed entry must actually change something, and a
        # non-disputed one must leave the stored label alone. Otherwise
        # `disputed` and `corrected_label` could drift apart and the
        # correction would be applied silently or not at all.
        if entry["disputed"]:
            assert entry["corrected_label"] != entry["stored_label"], (
                q_number
            )
        else:
            assert entry["corrected_label"] == entry["stored_label"], (
                q_number
            )

        assert entry["finding"].strip(), q_number
        assert entry["failure_mode"], q_number

    assert corrected_label(58, "A") == "D"
    assert corrected_label(54, "A") == "A"
    assert corrected_label(61, "A") == "A", "unlisted must pass through"
    assert disputed_questions() == [58]


if __name__ == "__main__":

    self_check()

    print("label audit self-check passed\n")

    for q_number in sorted(LABEL_AUDIT):

        entry = LABEL_AUDIT[q_number]

        print(
            f"Q{q_number}  {entry['status']}  "
            f"stored={entry['stored_label']} "
            f"corrected={entry['corrected_label']} "
            f"system={entry['system_answer']}  "
            f"[{entry['failure_mode']}]"
        )

    print(f"\ncorroborated: {list(CORROBORATED)}")
    print(f"unaudited:    {list(UNAUDITED)}")
    print(f"disputed:     {disputed_questions()}")
