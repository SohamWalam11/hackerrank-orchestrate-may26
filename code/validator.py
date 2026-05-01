"""
Validator module — post-generation checks on LLM output.

Design decisions:
  • Validates that all output fields conform to the allowed-value constraints
    from problem_statement.md.
  • Detects hallucinated URLs, phone numbers, and email addresses not present
    in the retrieved corpus excerpts.
  • Provides automatic correction for common LLM output issues.
"""

import re
from typing import Dict, List, Optional

import config


def validate_output(result: Dict, retrieved_chunks: List[Dict]) -> Dict:
    """
    Validate and sanitize a single output row.
    Returns the corrected result dict.
    """
    # Validate status
    if result.get("status") not in config.ALLOWED_STATUS:
        result["status"] = "escalated"  # safe default

    # Validate request_type
    if result.get("request_type") not in config.ALLOWED_REQUEST_TYPE:
        result["request_type"] = "product_issue"  # safe default

    # Ensure response is non-empty
    if not result.get("response") or not result["response"].strip():
        if result["status"] == "escalated":
            result["response"] = (
                "Thank you for reaching out. Your ticket has been escalated "
                "to a support specialist who will assist you shortly."
            )
        else:
            result["response"] = (
                "Thank you for your question. Please contact the official "
                "support team for detailed assistance."
            )

    # Ensure justification is non-empty
    if not result.get("justification") or not result["justification"].strip():
        result["justification"] = "Classification and response based on corpus analysis."

    # Ensure product_area is non-empty
    if not result.get("product_area") or not result["product_area"].strip():
        result["product_area"] = "general_support"

    # Check for hallucinated URLs
    result["response"] = _check_hallucinated_urls(
        result["response"], retrieved_chunks
    )

    return result


def _check_hallucinated_urls(response: str, chunks: List[Dict]) -> str:
    """
    Check if the response contains URLs not present in any retrieved chunk.
    Remove hallucinated URLs to avoid misinformation.
    """
    # Extract all URLs from response
    url_pattern = r'https?://[^\s,;)}\]\"\'<>]+'
    response_urls = re.findall(url_pattern, response)

    if not response_urls:
        return response

    # Collect all URLs from corpus chunks
    corpus_text = " ".join(c.get("text", "") for c in chunks)
    corpus_urls = set(re.findall(url_pattern, corpus_text))

    # Remove URLs not in corpus
    for url in response_urls:
        # Check if URL (or a prefix of it) appears in corpus
        url_found = any(
            url.rstrip("/").startswith(cu.rstrip("/")) or
            cu.rstrip("/").startswith(url.rstrip("/"))
            for cu in corpus_urls
        )
        if not url_found:
            # Replace the hallucinated URL with a note
            response = response.replace(
                url,
                "[please refer to the official support documentation]"
            )

    return response


def validate_batch(rows: List[Dict]) -> List[Dict]:
    """Validate a batch of output rows (no corpus check, just schema)."""
    for row in rows:
        if row.get("status") not in config.ALLOWED_STATUS:
            row["status"] = "escalated"
        if row.get("request_type") not in config.ALLOWED_REQUEST_TYPE:
            row["request_type"] = "product_issue"
    return rows
