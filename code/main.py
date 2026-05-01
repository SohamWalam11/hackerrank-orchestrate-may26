"""
Main entry point — CLI interface, batch CSV processing, and summary output.

Design decisions:
  • Loads .env automatically via python-dotenv (if present).
  • Accepts --input and --output CLI arguments, defaulting to config.py paths.
  • tqdm progress bar for visibility during batch runs.
  • Prints summary statistics after processing.
  • Seeds all random number generators for reproducibility.
  • The output CSV includes the input columns (issue, subject, company) echoed
    back, matching the output.csv schema from the reference repo.
"""

import argparse
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Load .env before importing config
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import config
from agent import SupportAgent

# Reproducibility
random.seed(config.SEED)
np.random.seed(config.SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Support Triage Agent — classify and respond to support tickets.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(config.INPUT_CSV),
        help=f"Path to input CSV (default: {config.INPUT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.OUTPUT_CSV),
        help=f"Path to output CSV (default: {config.OUTPUT_CSV})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for detailed output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # Load input CSV
    logger.info("Loading tickets from %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded %d tickets.", len(df))

    # Initialize agent
    logger.info("Initializing SupportAgent (loading corpus)...")
    start_time = time.time()
    agent = SupportAgent()
    init_time = time.time() - start_time
    logger.info(
        "Agent initialized in %.1fs — %d corpus chunks indexed.",
        init_time, len(agent.retriever.chunks),
    )

    # Process each ticket
    output_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing tickets"):
        ticket_dict = row.to_dict()
        ticket_start = time.time()
        result = agent.process_ticket(ticket_dict)
        ticket_time = time.time() - ticket_start

        # Echo input columns into output (matches output.csv schema)
        output_row = {
            "issue": ticket_dict.get("Issue", ticket_dict.get("issue", "")),
            "subject": ticket_dict.get("Subject", ticket_dict.get("subject", "")),
            "company": ticket_dict.get("Company", ticket_dict.get("company", "")),
            "response": result.get("response", ""),
            "product_area": result.get("product_area", ""),
            "status": result.get("status", "escalated"),
            "request_type": result.get("request_type", "product_issue"),
            "justification": result.get("justification", ""),
        }
        output_rows.append(output_row)

        logger.debug(
            "Ticket %d processed in %.1fs: status=%s, type=%s",
            idx, ticket_time, result["status"], result["request_type"],
        )

    # Build output DataFrame
    output_df = pd.DataFrame(output_rows)

    # Ensure column order matches output.csv header
    column_order = [
        "issue", "subject", "company", "response",
        "product_area", "status", "request_type", "justification",
    ]
    output_df = output_df[column_order]

    # Write output CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info("Output written to %s", output_path)

    # Print summary
    total_time = time.time() - start_time
    status_counts = Counter(output_df["status"])
    type_counts = Counter(output_df["request_type"])

    print("\n" + "=" * 50)
    print("  AI SUPPORT TRIAGE AGENT — SUMMARY")
    print("=" * 50)
    print(f"  Total tickets processed: {len(output_df)}")
    print(f"  Total time: {total_time:.1f}s ({total_time/max(len(output_df),1):.1f}s/ticket)")
    print(f"\n  Status breakdown:")
    print(f"    Replied:     {status_counts.get('replied', 0)}")
    print(f"    Escalated:   {status_counts.get('escalated', 0)}")
    print(f"\n  Request type breakdown:")
    for req_type, count in sorted(type_counts.items()):
        print(f"    {req_type}: {count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
