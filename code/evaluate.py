"""
Evaluation / cross-verification script.

Runs the agent against sample_support_tickets.csv (which has expected outputs)
and scores the predictions against ground truth. This lets you iterate on
the agent without burning submission attempts.

Usage:
  python code/evaluate.py                    # uses defaults
  python code/evaluate.py --verbose          # show per-ticket details
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # pylint: disable=wrong-import-position
from agent import SupportAgent  # pylint: disable=wrong-import-position

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate agent against sample tickets.")
    parser.add_argument("--verbose", action="store_true", help="Show per-ticket details.")
    return parser.parse_args()


def normalize(val):
    """Normalize a string for comparison."""
    if pd.isna(val) or val is None:
        return ""
    return str(val).strip().lower()


def score_status(predicted, expected):
    """Score status: exact match = 1, else 0."""
    return 1.0 if normalize(predicted) == normalize(expected) else 0.0


def score_request_type(predicted, expected):
    """Score request_type: exact match = 1, else 0."""
    return 1.0 if normalize(predicted) == normalize(expected) else 0.0


def score_product_area(predicted, expected):
    """Score product_area: exact match = 1, partial = 0.5, else 0."""
    p, e = normalize(predicted), normalize(expected)
    if p == e:
        return 1.0
    # Partial credit if one contains the other
    if p in e or e in p:
        return 0.5
    return 0.0


def score_response_quality(predicted, expected, status_pred, status_exp):
    """
    Score response quality:
    - If both escalated: check that response acknowledges escalation = 0.7
    - If both replied: check for keyword overlap with expected = variable
    - Mismatched status: 0
    """
    sp, se = normalize(status_pred), normalize(status_exp)
    if sp != se:
        return 0.0

    pp, ep = normalize(predicted), normalize(expected)

    if sp == "escalated":
        # For escalations, just check it's a reasonable escalation message
        escalation_words = ["escalat", "specialist", "review", "team", "human"]
        if any(w in pp for w in escalation_words):
            return 0.7
        return 0.3

    # For replies, check keyword overlap
    if not ep:
        return 0.5

    pred_words = set(pp.split())
    exp_words = set(ep.split())

    if not exp_words:
        return 0.5

    overlap = pred_words & exp_words
    jaccard = len(overlap) / max(len(pred_words | exp_words), 1)

    # Bonus for containing key phrases from expected
    return min(1.0, jaccard * 2 + 0.2)


def main():  # pylint: disable=too-many-locals,too-many-statements
    """Main evaluation routine."""
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    sample_path = config.SAMPLE_CSV
    if not sample_path.exists():
        logger.error("Sample CSV not found: %s", sample_path)
        sys.exit(1)

    logger.info("Loading sample tickets from %s", sample_path)
    df = pd.read_csv(sample_path)
    logger.info("Loaded %d sample tickets.", len(df))

    # Check for expected output columns
    expected_cols = {"Response", "Product Area", "Status", "Request Type"}
    actual_cols = set(df.columns)
    if not expected_cols.issubset(actual_cols):
        logger.error(
            "Sample CSV missing expected columns: %s",
            expected_cols - actual_cols,
        )
        sys.exit(1)

    # Initialize agent
    logger.info("Initializing SupportAgent...")
    agent = SupportAgent()
    logger.info("Agent ready — %d corpus chunks.", len(agent.retriever.chunks))

    # Process and score
    scores = {
        "status": [],
        "request_type": [],
        "product_area": [],
        "response": [],
    }
    details = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        ticket = {
            "issue": row.get("Issue", ""),
            "subject": row.get("Subject", ""),
            "company": row.get("Company", ""),
        }
        result = agent.process_ticket(ticket)

        # Expected values
        exp_status = str(row.get("Status", ""))
        exp_type = str(row.get("Request Type", ""))
        exp_area = str(row.get("Product Area", ""))
        exp_response = str(row.get("Response", ""))

        # Score
        s_status = score_status(result["status"], exp_status)
        s_type = score_request_type(result["request_type"], exp_type)
        s_area = score_product_area(result["product_area"], exp_area)
        s_resp = score_response_quality(
            result["response"], exp_response, result["status"], exp_status
        )

        scores["status"].append(s_status)
        scores["request_type"].append(s_type)
        scores["product_area"].append(s_area)
        scores["response"].append(s_resp)

        detail = {
            "idx": idx,
            "issue_preview": str(ticket["issue"])[:80],
            "status_score": s_status,
            "type_score": s_type,
            "area_score": s_area,
            "resp_score": s_resp,
            "pred_status": result["status"],
            "exp_status": exp_status,
            "pred_type": result["request_type"],
            "exp_type": exp_type,
            "pred_area": result["product_area"],
            "exp_area": exp_area,
        }
        details.append(detail)

        if args.verbose:
            total = (s_status + s_type + s_area + s_resp) / 4
            marker = "PASS" if total >= 0.75 else "FAIL"
            print(
                f"  [{marker}] Ticket {idx}: "
                f"status={s_status:.0f} type={s_type:.0f} "
                f"area={s_area:.1f} resp={s_resp:.2f} "
                f"| {detail['issue_preview']}"
            )

    # Compute averages
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    for dim, vals in scores.items():
        avg = sum(vals) / max(len(vals), 1)
        print(f"  {dim:15s}: {avg:.2%}  ({sum(v == 1.0 for v in vals)}/{len(vals)} perfect)")

    overall = sum(
        sum(vals) for vals in scores.values()
    ) / max(sum(len(vals) for vals in scores.values()), 1)
    print(f"\n  {'OVERALL':15s}: {overall:.2%}")
    print("=" * 60)

    # Show mismatches
    mismatches = [d for d in details if d["status_score"] < 1.0]
    if mismatches:
        print(f"\n  Status mismatches ({len(mismatches)}):")
        for d in mismatches:
            print(
                f"    Ticket {d['idx']}: predicted={d['pred_status']} "
                f"expected={d['exp_status']} | {d['issue_preview']}"
            )

    type_mismatches = [d for d in details if d["type_score"] < 1.0]
    if type_mismatches:
        print(f"\n  Request type mismatches ({len(type_mismatches)}):")
        for d in type_mismatches:
            print(
                f"    Ticket {d['idx']}: predicted={d['pred_type']} "
                f"expected={d['exp_type']} | {d['issue_preview']}"
            )


if __name__ == "__main__":
    main()
