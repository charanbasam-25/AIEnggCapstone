"""The learner's subject and topic choices, independent of model prompts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Subject:
    name: str
    description: str
    icon: str
    available: bool = False


SUBJECTS = (
    Subject("Polity", "Constitution, rights and governance", "⚖", True),
    Subject("Geography", "Physical and Indian geography", "◉"),
    Subject("History", "India's past and national movement", "⌛"),
    Subject("Economy", "Economic concepts and development", "↗"),
    Subject("Environment", "Ecology and conservation", "♧"),
)


TOPICS = {
    "Fundamental Rights": (
        "Equality before law and equal protection of laws; Article 14",
        "Prohibition of discrimination; Article 15 and its qualifications",
        "Freedom of speech and expression and reasonable restrictions; Article 19",
        "Protection of life and personal liberty; Article 21",
        "Constitutional remedies and writs; Article 32",
    ),
    "Directive Principles of State Policy": (
        "Application and non-enforceability of Directive Principles; Article 37",
        "Organisation of village panchayats; Article 40",
        "Uniform civil code; Article 44",
        "Separation of judiciary from executive; Article 50",
        "Promotion of international peace and security; Article 51",
    ),
    "Fundamental Duties": (
        "Respect for the Constitution, National Flag and National Anthem; Article 51A",
        "Protection of the natural environment; Article 51A",
        "Scientific temper, humanism and spirit of inquiry; Article 51A",
        "Preservation of the heritage of composite culture; Article 51A",
        "Educational opportunities for children aged six to fourteen; Article 51A",
    ),
    "Preamble and constitutional features": (
        "Preamble: justice, liberty, equality and fraternity",
        "Preamble: sovereign socialist secular democratic republic",
        "Constitutional framework and limits on government powers",
    ),
    "Parliament of India": (
        "Composition of Parliament; Article 79",
        "Composition of the Council of States; Article 80",
        "Composition of the House of the People; Article 81",
        "Money Bills and special procedure; Articles 109 and 110",
    ),
    "The President of India": (
        "Election of the President; Articles 54 and 55",
        "Qualifications for election as President; Article 58",
        "Term of office and impeachment; Articles 56 and 61",
        "Council of Ministers to aid and advise the President; Article 74",
    ),
    "The Supreme Court of India": (
        "Establishment and constitution of the Supreme Court; Article 124",
        "Original jurisdiction of the Supreme Court; Article 131",
        "Advisory jurisdiction of the Supreme Court; Article 143",
        "Constitutional remedies and writs; Article 32",
    ),
    "Election Commission of India": (
        "Superintendence, direction and control of elections; Article 324",
        "Adult suffrage; Article 326",
        "Composition and appointment of the Election Commission; Article 324",
    ),
    "Federalism and the Seventh Schedule": (
        "Union List and State List; Article 246 and Seventh Schedule",
        "Concurrent List and legislative powers; Article 246",
        "Residuary powers of legislation; Article 248",
    ),
    "Local Government": (
        "Gram Sabha and constitution of Panchayats; Articles 243A and 243B",
        "Duration of Panchayats; Article 243E",
        "Constitution of Municipalities; Article 243Q",
        "State Election Commission; Article 243K",
    ),
    "Amendment of the Constitution": (
        "Procedure for amendment of the Constitution; Article 368",
        "State ratification requirements for constitutional amendments; Article 368",
    ),
    "Emergency Provisions": (
        "Proclamation of Emergency; Article 352",
        "Failure of constitutional machinery in States; Article 356",
        "Financial emergency; Article 360",
    ),
    "Citizenship": (
        "Citizenship at commencement of the Constitution; Article 5",
        "Parliament's power to regulate citizenship by law; Article 11",
    ),
}

FORMAT_LABELS = {
    "simple": "Direct questions",
    "statements": "Statement-based questions",
    "mixed": "Mixed practice",
}
DIFFICULTY_LABELS = {
    "easy": "Foundation",
    "medium": "Standard",
    "hard": "Challenging",
}


def topic_focus(topic: str, index: int) -> str:
    focuses = TOPICS.get(topic, (topic,))
    return focuses[index % len(focuses)]
