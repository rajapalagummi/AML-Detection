# AML Transaction Detection
## PySpark ML Pipeline + Graph Pattern Analysis on IBM Synthetic Financial Dataset

---

## Overview

Anti-money laundering detection in financial institutions faces a specific challenge that most fraud detection systems don't: the suspicious activity isn't always visible at the transaction level. A single transaction sending $5,000 from one account to another looks unremarkable. The same transaction becomes suspicious when it's part of a cycle — money moving through 8 accounts across 4 countries before returning to the origin — or a fan-out pattern where one account distributes funds to 16 recipients simultaneously.

This project builds a two-layer detection system. The first layer uses PySpark ML to classify individual transactions as suspicious based on behavioral features — transaction volume, currency switching, timing, and account-level activity patterns. The second layer uses graph analysis to identify structural money laundering patterns — cycles, fan-out, fan-in, and scatter-gather — that only become visible when transactions are analyzed as a network rather than individually.

The dataset is the IBM synthetic AML dataset, which contains labeled transactions with eight real-world laundering pattern types embedded in a stream of legitimate transactions.

---

## Eight Laundering Pattern Types in the Dataset

**Fan-Out:** One account distributes funds to many recipients simultaneously. Characteristic of placement — converting cash into the financial system through many small transfers.

**Fan-In:** Many accounts funnel funds into one. Characteristic of integration — consolidating laundered money before final extraction.

**Cycle:** Money moves through a chain of accounts and returns to the origin. Characteristic of layering — obscuring the money trail through circular transfers.

**Stack:** Sequential chain transfers, each slightly reducing the amount. Designed to break the audit trail across multiple hops.

**Bipartite:** Two groups of accounts transact exclusively between each other in an alternating pattern.

**Gather-Scatter:** Funds collected from many sources, held briefly, then redistributed — a combination of fan-in and fan-out.

**Scatter-Gather:** Funds distributed to intermediaries, which then all send to a single destination.

**Random:** Multi-hop chains with random intermediate accounts, designed to evade pattern-matching rules.

---

## Architecture

The pipeline runs in four stages.

**Stage 1 — Data Pipeline:** PySpark loads the raw CSV transaction files, enforces schema, casts types, and filters invalid records. Transactions with zero payment amount or null laundering labels are excluded.

**Stage 2 — Feature Engineering:** Fourteen features are computed per transaction using PySpark window functions. Account-level features are computed over the full transaction history — how many transactions an account has sent, its average and maximum transaction amounts, how many unique counterparties it has reached, and how many distinct currencies it has used. Transaction-level features capture cross-currency conversion, amount ratio between sent and received, and whether the transaction occurred during overnight hours.

**Stage 3 — ML Classification:** Two PySpark MLlib models are trained — Random Forest (100 trees, max depth 6) and Gradient Boosted Trees (50 iterations, max depth 5). Both use an 80/20 stratified split. Evaluation covers AUC, F1, precision, and recall. Precision and recall are computed directly from confusion matrix counts rather than inferred from aggregate metrics, which gives honest per-class performance.

**Stage 4 — Graph Pattern Detection:** Transactions are converted to a directed graph where nodes are accounts and edges are transactions. Fan-out and fan-in are detected by out-degree and in-degree thresholds. Cycles are detected using NetworkX's simple cycle enumeration with a maximum length cap to prevent combinatorial explosion on large graphs. Each detected pattern is associated with its laundering edge count.

---

## Key Results

Run `python3 main.py` on the HI-Small dataset to generate actual results. Typical results on this dataset:

- Class imbalance: laundering transactions represent approximately 2-5% of total volume
- Graph: thousands of accounts, hundreds of flagged structural patterns
- ML: AUC in the 0.85-0.95 range depending on feature quality

The full results with exact numbers are saved to `outputs/pipeline_summary.json` after each run.

---

## How to Run

```bash
# 1. Prerequisites — Java required for PySpark
# Install Java: brew install openjdk@17
# Verify: java -version

# 2. Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Place data files in data/ folder
# Required: HI-Small_Trans.csv, HI-Small_Accounts.csv, HI-Small_Patterns.txt
# Download from: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

# 4. Run full pipeline
python3 main.py

# 5. Skip ML training for faster graph analysis only
python3 main.py --skip-ml

# 6. Skip graph analysis for faster ML only
python3 main.py --skip-graph
```

---


*Built by Raja Palagummi | rajapalagummi.com | github.com/rajapalagummi*
