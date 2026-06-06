"""
AML Transaction Detection — IBM Synthetic Dataset
PySpark ML + Graph Pattern Analysis

Usage:
    python3 main.py
    python3 main.py --skip-ml       # skip model training (faster)
    python3 main.py --skip-graph    # skip graph analysis
"""
import os
import sys
import json
import argparse
from datetime import datetime

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = "outputs"

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   AML Transaction Detection                                      ║
║   IBM Synthetic Financial Dataset                                ║
║                                                                  ║
║   Stage 1: PySpark Data Pipeline                                 ║
║   Stage 2: Feature Engineering (transaction + graph features)    ║
║   Stage 3: ML Classification (Random Forest + GBT)              ║
║   Stage 4: Graph Pattern Detection (Fan-Out, Fan-In, Cycles)    ║
╚══════════════════════════════════════════════════════════════════╝
"""


def run(skip_ml=False, skip_graph=False):
    print(BANNER)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from src.data_pipeline import get_spark, load_transactions, load_accounts, summarize
    from src.feature_engineering import build_transaction_features

    print("=" * 60)
    print("  STAGE 1: Data Pipeline")
    print("=" * 60)

    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")

    trans    = load_transactions(spark, "data/HI-Small_Trans.csv")
    accounts = load_accounts(spark, "data/HI-Small_Accounts.csv")
    total, laundering = summarize(trans, "Transactions")

    print("\n  STAGE 2: Feature Engineering")
    print("=" * 60)
    trans_feat = build_transaction_features(trans)
    print(f"[Features] {len(__import__('src.feature_engineering', fromlist=['FEATURE_COLS']).FEATURE_COLS)} features engineered")

    ml_results = {}
    if not skip_ml:
        print("\n" + "=" * 60)
        print("  STAGE 3: ML Classification")
        print("=" * 60)
        from src.ml_classifier import train_and_evaluate
        ml_results = train_and_evaluate(trans_feat)

    graph_results = {}
    trans_pdf = None
    if not skip_graph:
        print("\n" + "=" * 60)
        print("  STAGE 4: Graph Pattern Detection")
        print("=" * 60)
        from src.graph_detector import run_graph_analysis
        trans_pdf = trans.select(
            "from_account", "to_account",
            "amount_paid", "payment_currency", "is_laundering"
        ).toPandas()
        graph_results = run_graph_analysis(trans_pdf)

    print("\n" + "=" * 60)
    print("  Generating Dashboard")
    print("=" * 60)

    summary = {
        "generated_at":            datetime.now().isoformat(),
        "dataset":                 "IBM AML Synthetic Dataset — HI-Small",
        "total_transactions":      int(total),
        "laundering_transactions": int(laundering),
        "laundering_pct":          round(laundering / total * 100, 2),
        "ml_results":              ml_results,
        "graph_results":           {
            k: v for k, v in graph_results.items()
            if k not in ["fan_out", "fan_in", "cycles"]
        },
    }

    from src.visualizations import generate_dashboard
    generate_dashboard(trans_pdf, ml_results, graph_results, summary)

    summary_path = os.path.join(OUTPUT_DIR, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    best_model = max(ml_results, key=lambda m: ml_results[m]["auc"]) if ml_results else "N/A"
    best = ml_results.get(best_model, {})

    print(f"""
{'='*60}
  ✓ PIPELINE COMPLETE
{'='*60}

📊 Key Results:
   Total transactions:    {total:,}
   Laundering:            {laundering:,} ({summary['laundering_pct']:.2f}%)
   Best model ({best_model}):
     AUC:                 {best.get('auc', 'N/A')}
     F1:                  {best.get('f1', 'N/A')}
     Precision:           {best.get('precision', 'N/A')}
     Recall:              {best.get('recall', 'N/A')}
   Graph:
     Flagged accounts:    {graph_results.get('flagged_accounts', 'N/A')}
     Cycles detected:     {graph_results.get('cycle_count', 'N/A')}

📁 Outputs:
   outputs/aml_dashboard.png
   outputs/pipeline_summary.json
""")

    spark.stop()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Detection Pipeline")
    parser.add_argument("--skip-ml",    action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    args = parser.parse_args()
    run(skip_ml=args.skip_ml, skip_graph=args.skip_graph)
