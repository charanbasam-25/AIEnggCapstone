# Project brief

## Purpose

UPSC Practice helps a learner practise static Indian Polity through source-grounded MCQs. The application generates or reuses questions, checks their answer and explanations before delivery, and shows sources after submission.

The capstone objective is measured educational usefulness: correct keys, defensible distractors, source-supported explanations, useful withholding, acceptable yield, latency and cost. A saved model verdict or a green workflow gate is not a substitute for independent expert review.

## Scope

Polity is open with 13 curated topics: Fundamental Rights, Directive Principles, Fundamental Duties, Preamble and constitutional features, Parliament, President, Supreme Court, Election Commission, federalism, local government, amendment procedure, emergency provisions and citizenship. Geography, History, Economy and Environment are visible but locked.

Current affairs, live office-holders, unrestricted web search, private uploads and open-ended tutoring are outside the current implementation. Static historical material still needs an appropriate source and date.

## Data and privacy

The current corpus has 1,149 page-aware chunks: 1,107 from the Constitution and 42 from selected NCERT content. Accepted questions and operational traces are stored in local SQLite. Answers, scores and the latest 20 sets are in Streamlit server memory for the active session. The app does not ask for names, email or free text, but it does not include a general PII scanner, login, user isolation, encryption-at-rest or retention controls. It is a local prototype, not a production privacy boundary.

## Architecture

![System architecture](diagrams/practice_architecture.png)

The editable diagrams and their renderer are in ARCHITECTURE_DIAGRAMS.md and diagrams/render_diagrams.py. The main flow is: extract page-aware source text; retrieve evidence before drafting; generate a structured candidate; verify claims or compare options; audit quality and write reviewed notes; accept, revise or withhold; save the accepted item and present it in a quiz.

## Guardrails

The five mandatory gates are sources, format, answer, quality and explanations. Source references are selected from a checked catalog and bound by Python to exact text and page metadata. Every revision clears previous results. The service revalidates the complete record before persistence. These controls reduce risk; they do not guarantee correctness.

## Evaluation contract

The project reports coverage, precision when answered, overall accuracy, error rate, abstentions, gate yield, grounding checks, retrieval hit rate, latency, calls, tokens and cost. The 75-question mixed set is a regression diagnostic. Generated-question correctness still needs an expert-labeled frozen benchmark of static questions with key, ambiguity, option-note and evidence labels.

## Operational limits

The model client uses a 45-second timeout and one SDK transport retry. A request slot has one initial draft and at most two revisions; explanation writing has one local repair. If accepted items exist when a later slot fails, the service can return a partial set. A source change retires the active set.
