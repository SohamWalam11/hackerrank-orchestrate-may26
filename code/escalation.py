"""
Escalation module — decides whether a ticket should be auto-replied or
escalated to a human agent.

Design decisions:
  • Multi-signal escalation: risk score, retrieval quality, company
    detection confidence, content complexity, and sensitive topic detection
    all feed into the decision.
  • Deterministic: no randomness, no LLM dependency — pure function.
  • Separate from the LLM call so we never generate a response for tickets
    that require human judgment.
"""

import re
from typing import List, Dict

import config


# PII / sensitive data patterns
_SENSITIVE_PATTERNS = [
    r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # credit card number
    r'\b\d{3}-\d{2}-\d{4}\b',                          # SSN
    r'\b[A-Z]{2}\d{7}\b',                               # passport-like
]


def should_escalate(
    issue: str,
    risk: float,
    retrieved_chunks: List[Dict],
    company_confidence: float = 1.0,
    request_type: str = "product_issue",
) -> tuple:
    """
    Return (should_escalate: bool, reason: str) for the ticket.

    Multi-signal decision:
      1. risk_score > RISK_THRESHOLD → escalate (sensitive/dangerous)
      2. No retrieved chunks → escalate (no corpus coverage)
      3. Best chunk score <= RETRIEVAL_MIN_SCORE → escalate (low confidence)
      4. Company detection confidence < CONFIDENCE_THRESHOLD → escalate
      5. Contains sensitive PII patterns → escalate
      6. Prompt injection detected → escalate (reply with out-of-scope)
      7. Otherwise → reply
    """
    reasons = []

    # 1. High risk score
    if risk > config.RISK_THRESHOLD:
        reasons.append(
            f"High risk score ({risk:.2f}) — ticket involves sensitive/dangerous content "
            f"requiring human review."
        )

    # 2. No retrieved chunks
    if len(retrieved_chunks) == 0:
        reasons.append(
            "No relevant documentation found in the support corpus."
        )

    # 3. Low retrieval confidence
    if retrieved_chunks:
        best_score = max(c["score"] for c in retrieved_chunks)
        if best_score <= config.RETRIEVAL_MIN_SCORE:
            reasons.append(
                f"Low retrieval confidence (best score: {best_score:.4f}) — "
                f"corpus may not cover this topic adequately."
            )

    # 4. Company detection uncertainty
    if company_confidence < config.CONFIDENCE_THRESHOLD:
        reasons.append(
            f"Company detection confidence too low ({company_confidence:.2f}) — "
            f"risk of routing to wrong product ecosystem."
        )

    # 5. Sensitive PII detected
    for pattern in _SENSITIVE_PATTERNS:
        if re.search(pattern, issue):
            reasons.append(
                "Sensitive personal information (PII) detected in ticket — "
                "requires human handling for privacy compliance."
            )
            break

    # 6. Prompt injection → we don't escalate, we reply with out-of-scope
    # (handled in agent.py — injection → invalid → reply with rejection)

    if reasons:
        return True, " | ".join(reasons)

    return False, ""
