# AI Support Triage Agent

A terminal-based AI agent that triages real support tickets across three product ecosystems — **HackerRank**, **Claude**, and **Visa** — using only the provided support corpus.

## Architecture

```
┌───────────────────────────────────────────────────────┐
│                    main.py (CLI)                      │
│  Loads CSV → Iterates tickets → Writes output.csv     │
└──────────────────────┬────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────┐
│               agent.py (SupportAgent)                 │
│  Orchestrates the full pipeline per ticket:           │
│  classify → retrieve → escalate check → LLM gen      │
└───┬──────────┬──────────┬──────────┬──────────┬───────┘
    │          │          │          │          │
    ▼          ▼          ▼          ▼          ▼
 config.py  classifier  retriever  escalation  validator
            .py         .py        .py         .py
```

### Module Responsibilities

| Module | Role |
|--------|------|
| `config.py` | Centralized paths, model config, thresholds, env vars |
| `retriever.py` | Hybrid BM25 + TF-IDF corpus retriever with sentence-aware chunking & RRF |
| `classifier.py` | Company detection (with confidence), request type, weighted risk scoring |
| `escalation.py` | Multi-signal escalation decision (risk, retrieval quality, PII, confidence) |
| `prompts.py` | Chain-of-thought prompts with few-shot examples, anti-hallucination guardrails |
| `validator.py` | Post-generation validation: schema checks, hallucinated URL detection |
| `agent.py` | SupportAgent class — orchestrates the full pipeline per ticket |
| `main.py` | CLI entry point — batch CSV processing with timing and summary |

## Setup

```bash
cd code
pip install -r requirements.txt
```

Set your Anthropic API key:

```bash
# Linux/Mac
export ANTHROPIC_API_KEY=your_key_here

# Windows PowerShell
$env:ANTHROPIC_API_KEY="your_key_here"

# Or create a .env file in the project root:
echo "ANTHROPIC_API_KEY=your_key_here" > ../.env
```

## Usage

```bash
# Default: process support_tickets/support_tickets.csv → support_tickets/output.csv
python main.py

# Custom paths
python main.py --input /path/to/input.csv --output /path/to/output.csv

# Verbose debug logging
python main.py --verbose
```

## Pipeline (Per Ticket)

1. **Company Detection** — Keyword heuristics with confidence score. Trusts `company` field when present, validates with keyword evidence.
2. **Request Type Classification** — Heuristic first-pass (bug/feature_request/invalid/product_issue), refined by LLM with few-shot examples.
3. **Corpus Retrieval** — Hybrid BM25 + TF-IDF with Reciprocal Rank Fusion. 800-char sentence-aware chunks, company-filtered with fallback.
4. **Risk Assessment** — Weighted keyword scoring for fraud, legal, PII, prompt injection detection.
5. **Escalation Decision** — Multi-signal: risk score, retrieval quality, company confidence, PII patterns.
6. **LLM Response Generation** — Claude (claude-sonnet-4-20250514, temp=0) with chain-of-thought prompting, strict corpus grounding, and few-shot examples.
7. **Post-Generation Validation** — Schema enforcement, hallucinated URL detection, safe defaults.

## Output Schema

| Column | Description |
|--------|-------------|
| `issue` | Original issue text (echoed) |
| `subject` | Original subject (echoed) |
| `company` | Original company (echoed) |
| `response` | User-facing answer grounded in corpus |
| `product_area` | Support category (e.g. "screen", "billing", "privacy") |
| `status` | `"replied"` or `"escalated"` |
| `request_type` | `"product_issue"`, `"feature_request"`, `"bug"`, or `"invalid"` |
| `justification` | Explanation of the decision, citing corpus |

## Key Design Decisions

- **Hybrid retrieval (BM25 + TF-IDF)**: BM25 handles keyword relevance well; TF-IDF captures n-gram similarity. RRF merges without score calibration.
- **Sentence-aware chunking**: Avoids splitting mid-sentence, improving excerpt readability.
- **Deterministic pipeline**: Temperature=0, seeded RNG, keyword-based escalation — reproducible results.
- **Multi-signal escalation**: Never guesses on high-risk tickets — fraud, legal, PII, or low-confidence retrieval all trigger escalation.
- **Anti-hallucination**: Prompts enforce corpus grounding; validator catches fabricated URLs.
- **Graceful degradation**: Every step has try/except with safe fallbacks. The batch never crashes on a single bad ticket.

## Escalation Criteria

Tickets are escalated when:
- Risk score > 0.55 (fraud, legal, identity theft, etc.)
- No relevant corpus chunks found
- Best retrieval score ≤ 0.08 (low confidence)
- Company detection confidence < 0.3
- Sensitive PII patterns detected (card numbers, SSNs)
