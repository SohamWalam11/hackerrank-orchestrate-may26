"""
Agent module — the core SupportAgent class that orchestrates retrieval,
classification, escalation decisions, and Claude API calls.

Design decisions:
  • Multi-pass pipeline: classify → retrieve → escalation check → generate.
  • The LLM is used for both classification refinement and response generation.
  • JSON responses are extracted with a robust regex that handles nested objects,
    markdown code fences, and surrounding text.
  • Every ticket processing step is wrapped in try/except so batch runs
    never crash on a single bad row.
  • Post-generation validation catches hallucinated content.
"""

import json
import re
import time
import logging
from typing import Dict, List, Optional

import anthropic

import config
import prompts
from retriever import CorpusRetriever
from classifier import detect_company, classify_request_type, risk_score
from escalation import should_escalate
from validator import validate_output

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> Optional[Dict]:
    """
    Extract a JSON object from Claude's response.
    Handles markdown code fences, nested objects, and surrounding text.
    Returns None if parsing fails.
    """
    if not text:
        return None

    # Strip markdown code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # Try to find a complete JSON object — handle nested braces
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
                    continue

    # Fallback: regex for simple object
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _call_claude(
    client: anthropic.Anthropic,
    system: str,
    user_message: str,
    max_retries: int = 3,
    max_tokens: int = 1024,
) -> str:
    """
    Call the Claude API with exponential backoff on failure.
    Returns the text content of the first block in the response.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=config.MODEL,
                max_tokens=max_tokens,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "Claude API call failed (attempt %d/%d): %s. Retrying in %ds...",
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)

    logger.error("All %d Claude API retries exhausted.", max_retries)
    raise last_exc


# ---------------------------------------------------------------------------
# SupportAgent
# ---------------------------------------------------------------------------

class SupportAgent:
    """
    High-level support triage agent.

    Pipeline per ticket:
      1. Detect company (heuristic + confidence score).
      2. Classify request type (heuristic + LLM refinement).
      3. Retrieve relevant corpus chunks (hybrid BM25 + TF-IDF).
      4. Compute risk score and make escalation decision.
      5. For replies: call Claude to generate a grounded response.
      6. For escalations: call Claude to generate an empathetic escalation msg.
      7. Validate output (schema, hallucination check).
      8. Return a dict with all required output columns.
    """

    def __init__(self):
        if not config.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it with: export ANTHROPIC_API_KEY=your_key_here"
            )
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.retriever = CorpusRetriever()
        logger.info(
            "SupportAgent initialized — %d corpus chunks indexed.",
            len(self.retriever.chunks),
        )

    def process_ticket(self, row: Dict) -> Dict:
        """
        Process a single support ticket row.
        Returns a dict with keys:
          status, product_area, response, justification, request_type

        Wrapped in try/except so a single malformed row never crashes the batch.
        """
        try:
            return self._process_ticket_safe(row)
        except Exception as exc:
            logger.error(
                "Error processing ticket (subject=%s): %s",
                row.get("subject", "unknown"), exc,
            )
            return self._safe_fallback(
                row, reason=f"Processing error: {exc}"
            )

    def _process_ticket_safe(self, row: Dict) -> Dict:
        """Core ticket processing pipeline."""
        issue = str(row.get("issue", "") or "").strip()
        subject = str(row.get("subject", "") or "").strip()
        stated_company = str(row.get("company", "") or "").strip()

        # Step 1: Detect company
        company, company_conf = detect_company(issue, subject, stated_company)
        logger.debug("Company: %s (conf: %.2f)", company, company_conf)

        # Step 2: Initial request type classification (heuristic)
        req_type = classify_request_type(issue)
        logger.debug("Request type (heuristic): %s", req_type)

        # Step 3: Retrieve relevant corpus chunks
        search_company = company if company != "Unknown" else None
        chunks = self.retriever.retrieve(query=issue, company=search_company)
        logger.debug("Retrieved %d chunks (best score: %.4f)",
                      len(chunks),
                      max((c["score"] for c in chunks), default=0))

        # Step 4: Special handling for clearly invalid/out-of-scope
        # Must happen BEFORE escalation check — otherwise low-confidence
        # retrieval or company detection escalates what should be a simple
        # "out of scope" reply.
        if req_type == "invalid":
            return self._handle_invalid(row, issue, subject, company, chunks)

        # Step 5: Risk assessment
        risk = risk_score(issue)
        logger.debug("Risk score: %.2f", risk)

        # Step 6: Escalation decision
        escalate, esc_reason = should_escalate(
            issue, risk, chunks, company_conf, req_type
        )

        # Step 6: LLM-refined classification (if we have chunks)
        if chunks and not escalate:
            llm_classification = self._llm_classify(
                issue, subject, company, chunks
            )
            if llm_classification:
                if llm_classification.get("request_type") in config.ALLOWED_REQUEST_TYPE:
                    req_type = llm_classification["request_type"]

        # Step 7: Generate response
        if escalate:
            result = self._handle_escalation(
                row, issue, subject, company, chunks, risk, esc_reason, req_type
            )
        else:
            result = self._handle_reply(
                row, issue, subject, company, req_type, chunks
            )

        # Step 8: Validate
        result = validate_output(result, chunks)

        return result

    # -------------------------------------------------------------------
    # Response handlers
    # -------------------------------------------------------------------

    def _handle_reply(
        self, row: Dict, issue: str, subject: str,
        company: str, req_type: str, chunks: List[Dict],
    ) -> Dict:
        """Generate an auto-reply using Claude."""
        corpus_context = self._build_corpus_context(chunks)

        user_msg = prompts.REPLY_PROMPT.format(
            company=company,
            subject=subject,
            issue=issue,
            corpus_excerpts=corpus_context,
            few_shot_section=prompts.FEW_SHOT_REPLY_EXAMPLES,
        )

        try:
            raw = _call_claude(self.client, prompts.SYSTEM_PROMPT, user_msg)
            parsed = _parse_json_response(raw)
            if parsed and "response" in parsed:
                return {
                    "status": "replied",
                    "product_area": parsed.get("product_area", "general_support"),
                    "response": parsed["response"],
                    "justification": parsed.get("justification", "Response grounded in corpus excerpts."),
                    "request_type": req_type,
                }
        except Exception as exc:
            logger.warning("Claude call failed during reply: %s", exc)

        # Fallback: escalate if LLM fails
        return self._safe_fallback(
            row, reason="LLM response generation failed", req_type=req_type,
        )

    def _handle_escalation(
        self, row: Dict, issue: str, subject: str,
        company: str, chunks: List[Dict], risk: float,
        esc_reason: str, req_type: str,
    ) -> Dict:
        """Generate an escalation message using Claude."""
        corpus_context = self._build_corpus_context(chunks)

        user_msg = prompts.ESCALATE_PROMPT.format(
            company=company,
            subject=subject,
            issue=issue,
            escalation_reason=esc_reason,
            corpus_excerpts=corpus_context,
        )

        try:
            raw = _call_claude(self.client, prompts.SYSTEM_PROMPT, user_msg)
            parsed = _parse_json_response(raw)
            if parsed and "response" in parsed:
                return {
                    "status": "escalated",
                    "product_area": parsed.get("product_area", "general_support"),
                    "response": parsed["response"],
                    "justification": parsed.get("justification", esc_reason),
                    "request_type": req_type,
                }
        except Exception as exc:
            logger.warning("Claude call failed during escalation: %s", exc)

        # Deterministic fallback
        return {
            "status": "escalated",
            "product_area": "general_support",
            "response": (
                f"Thank you for contacting {company} support. "
                f"Your issue has been escalated to a specialist who will "
                f"review your case and respond within 24-48 hours."
            ),
            "justification": esc_reason or "Escalated based on risk and retrieval analysis.",
            "request_type": req_type,
        }

    def _handle_invalid(
        self, row: Dict, issue: str, subject: str,
        company: str, chunks: List[Dict],
    ) -> Dict:
        """Handle clearly invalid or out-of-scope tickets."""
        # Check if it's a simple greeting/thank you
        lower = issue.lower()
        greetings = ["thank", "thanks", "happy to help", "bye", "goodbye"]
        if any(g in lower for g in greetings):
            return {
                "status": "replied",
                "product_area": "conversation_management",
                "response": "Happy to help! Let us know if you have any other questions.",
                "justification": "Simple greeting/thank-you message — no action needed.",
                "request_type": "invalid",
            }

        return {
            "status": "replied",
            "product_area": "conversation_management",
            "response": (
                "I am sorry, this is out of scope from my capabilities. "
                "I can only assist with HackerRank, Claude, or Visa support questions."
            ),
            "justification": "Issue is unrelated to any supported product ecosystem.",
            "request_type": "invalid",
        }

    # -------------------------------------------------------------------
    # LLM-assisted classification
    # -------------------------------------------------------------------

    def _llm_classify(
        self, issue: str, subject: str, company: str, chunks: List[Dict],
    ) -> Optional[Dict]:
        """Use Claude for refined classification when heuristics are ambiguous."""
        corpus_context = self._build_corpus_context(chunks[:3])  # fewer for speed

        user_msg = prompts.CLASSIFY_PROMPT.format(
            company=company,
            subject=subject,
            issue=issue,
            corpus_excerpts=corpus_context,
        )

        try:
            raw = _call_claude(
                self.client, prompts.SYSTEM_PROMPT, user_msg, max_tokens=256
            )
            return _parse_json_response(raw)
        except Exception as exc:
            logger.debug("LLM classification failed: %s", exc)
            return None

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _build_corpus_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into a readable context string."""
        if not chunks:
            return "No relevant corpus excerpts found."
        lines = []
        for i, chunk in enumerate(chunks, 1):
            topic = chunk.get("topic", "")
            topic_str = f" | topic: {topic}" if topic else ""
            lines.append(
                f"[Excerpt {i}] (source: {chunk['source']}{topic_str}, "
                f"score: {chunk['score']})"
            )
            lines.append(chunk["text"])
            lines.append("")
        return "\n".join(lines)

    def _safe_fallback(
        self, row: Dict,
        reason: str = "Unknown error during processing.",
        req_type: str = "product_issue",
    ) -> Dict:
        """Return a safe escalation dict that never crashes the batch."""
        stated = str(row.get("company", "") or "Unknown")
        return {
            "status": "escalated",
            "product_area": "general_support",
            "response": (
                "Thank you for reaching out. Your ticket has been escalated "
                "to a support specialist who will assist you shortly."
            ),
            "justification": reason,
            "request_type": req_type,
        }
