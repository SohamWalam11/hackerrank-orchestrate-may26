"""
Classifier module — heuristics for company detection, request-type
classification, and risk scoring.

Design decisions:
  • Keyword-based rules provide fast, deterministic first-pass classification.
  • Confidence scores allow the agent to fall back to LLM classification
    when heuristics are uncertain.
  • Risk scoring uses weighted keywords — some signals (fraud, legal) are
    worth more than others (refund, dispute).
"""

import re
from typing import Tuple

# ---------------------------------------------------------------------------
# Company keywords — ordered by specificity
# ---------------------------------------------------------------------------
_COMPANY_KEYWORDS = {
    "hackerrank": [
        "hackerrank", "hacker rank", "codepair", "codescreen",
        "certified skill", "skill certification", "proctor", "plagiarism",
        "custom test", "test suite", "coding test", "developer assessment",
        "assessment", "codepair interview", "hackerrank for work",
        "test variant", "candidate", "interview pad", "mock interview",
        "skillup", "code challenge", "hiring test", "recruiter",
    ],
    "claude": [
        "claude", "anthropic", "claude ai", "claude api", "claude pro",
        "anthropic api", "message limit", "claude max", "claude team",
        "claude enterprise", "artifacts", "claude code", "claude.ai",
        "bedrock", "aws bedrock",
    ],
    "visa": [
        "visa", "visanet", "visa developer", "visa api",
        "visa direct", "cybersource", "visa checkout",
        "visa card", "visa debit", "visa credit", "visa prepaid",
        "traveller's cheque", "travelers cheque", "traveler's cheque",
        "chargeback", "merchant", "cardholder", "card stolen",
        "lost card", "visa india",
    ],
}

# ---------------------------------------------------------------------------
# Risk keywords with severity weights
# ---------------------------------------------------------------------------
_RISK_KEYWORDS = {
    # High severity (weight 3)
    "fraud": 3, "fraudulent": 3, "identity theft": 3, "stolen": 3,
    "hacked": 3, "compromised": 3, "data breach": 3, "data leak": 3,
    "security vulnerability": 3, "breach": 3, "exposed": 3, "leaked": 3,

    # Medium severity (weight 2)
    "legal": 2, "lawsuit": 2, "lawyer": 2, "attorney": 2,
    "sue": 2, "court": 2, "unauthorized": 2, "abuse": 2,
    "harass": 2, "threat": 2, "discrimination": 2,
    "gdpr": 2, "ccpa": 2, "account locked": 2, "locked out": 2,

    # Lower severity (weight 1)
    "dispute": 1, "refund": 1, "chargeback": 1,
    "pii": 1, "personal information": 1,
    "delete my account": 1, "delete all": 1,
}

# ---------------------------------------------------------------------------
# Request type keywords
# ---------------------------------------------------------------------------
_BUG_KEYWORDS = [
    "bug", "broken", "not working", "error", "crash", "fail", "fails",
    "cannot", "can't", "unable", "doesn't work", "does not work",
    "500", "502", "503", "timeout", "down", "not loading",
    "failing", "stopped working", "not responding", "blocker",
]

_FEATURE_KEYWORDS = [
    "feature", "would be nice", "wish", "suggest", "enhancement",
    "improvement", "add support", "please add", "new feature",
    "could you add", "it would be great", "would love",
]

_INVALID_KEYWORDS = [
    "spam", "test message", "asdf", "nonsense", "gibberish", "junk",
    "iron man", "movie", "actor", "football", "cricket", "weather",
    "recipe", "song", "lyrics",
]

# Prompt injection / manipulation patterns
_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+(instructions|rules|prompts)",
    r"(display|show|reveal|dump)\s+(all|the|your)\s+(rules|instructions|internal|system|logic|prompt)",
    r"pretend\s+you\s+are",
    r"you\s+are\s+now\s+in\s+.*mode",
    r"delete\s+all\s+files",
    r"format\s+(?:c|the)\s*(?:drive|disk)",
    r"(?:rm|del)\s+-rf",
]


# ---------------------------------------------------------------------------
# Company detection
# ---------------------------------------------------------------------------

def detect_company(
    issue: str,
    subject: str,
    stated_company: str = "",
) -> Tuple[str, float]:
    """
    Detect the company from issue text, subject, and the stated_company field.
    Returns (company_name, confidence) where confidence ∈ [0, 1].
    Company is one of: "HackerRank", "Claude", "Visa", "Unknown".
    """
    text = f"{subject} {issue}".lower()

    # Score each company by keyword hits
    scores = {}
    for company, keywords in _COMPANY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        scores[company] = score

    total_hits = sum(scores.values())

    # If a company is explicitly stated, validate it
    if stated_company and stated_company.strip().lower() != "none":
        stated = stated_company.strip().lower()
        for company in _COMPANY_KEYWORDS:
            if stated == company or company in stated or stated in company:
                # Trust the stated company, with boosted confidence if keywords match
                keyword_conf = min(1.0, scores.get(company, 0) / 3.0)
                return company.title(), max(0.7, keyword_conf)

    # No stated company or stated "None" — pick best by keyword count
    if total_hits == 0:
        return "Unknown", 0.0

    best_company = max(scores, key=scores.get)
    best_score = scores[best_company]
    confidence = min(1.0, best_score / max(total_hits, 1))

    return best_company.title(), confidence


# ---------------------------------------------------------------------------
# Request type classification
# ---------------------------------------------------------------------------

def classify_request_type(issue: str) -> str:
    """
    Classify the issue into one of:
      "bug" | "feature_request" | "invalid" | "product_issue"
    Checked in priority order: invalid → bug → feature_request → product_issue.
    """
    text = issue.lower()

    # Check for prompt injection / manipulation first
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text):
            return "invalid"

    # Check for clearly invalid / out-of-scope content
    if _is_out_of_scope(text):
        return "invalid"

    if any(kw in text for kw in _INVALID_KEYWORDS):
        return "invalid"

    # Bug indicators
    bug_hits = sum(1 for kw in _BUG_KEYWORDS if kw in text)
    feature_hits = sum(1 for kw in _FEATURE_KEYWORDS if kw in text)

    if bug_hits > feature_hits and bug_hits > 0:
        return "bug"
    if feature_hits > 0:
        return "feature_request"

    return "product_issue"


def _is_out_of_scope(text: str) -> bool:
    """Check if the text is clearly unrelated to HackerRank/Claude/Visa."""
    words = text.split()

    # Pure greetings / pleasantries — regardless of word count
    greeting_phrases = [
        "thank you for helping",
        "thanks for helping",
        "thank you for your help",
        "thanks for your help",
        "happy to help",
        "thanks a lot",
        "thank you so much",
    ]
    if any(g in text for g in greeting_phrases):
        return True

    # Short messages (up to 8 words) with no relevant product keywords
    if len(words) <= 8:
        all_keywords = set()
        for kws in _COMPANY_KEYWORDS.values():
            all_keywords.update(kws)
        if not any(kw in text for kw in all_keywords):
            greetings = [
                "thank", "thanks", "hello", "hi ", "hey", "bye",
                "goodbye", "cheers", "good morning", "good evening",
            ]
            if any(g in text for g in greetings):
                return True
    return False


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

def risk_score(issue: str) -> float:
    """
    Return a float in [0, 1] indicating how risky the issue is.
    0 = completely safe, 1 = extremely risky (fraud, legal, etc.).
    Uses weighted keywords for more nuanced scoring.
    """
    text = issue.lower()

    # Weighted keyword scoring
    total_weight = sum(
        weight for kw, weight in _RISK_KEYWORDS.items() if kw in text
    )

    # Check for prompt injection — always high risk
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text):
            total_weight += 5

    # Normalize: max realistic weight ~15, map to [0, 1]
    if total_weight == 0:
        return 0.0
    return min(1.0, 0.25 + 0.1 * total_weight)
