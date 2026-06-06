import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

OUTPUT_DIR = "outputs"


def generate_dashboard(trans_pdf, ml_results, graph_results, summary):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "AML Transaction Detection — IBM Synthetic Dataset\nPySpark ML + Graph Pattern Analysis",
        fontsize=15, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    BLUE   = "#2E75B6"
    RED    = "#C93828"
    GREEN  = "#0A8F5C"
    ORANGE = "#B87200"

    # Panel 1: Class distribution
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ["Legitimate", "Laundering"]
    counts = [
        summary["total_transactions"] - summary["laundering_transactions"],
        summary["laundering_transactions"]
    ]
    ax1.bar(labels, counts, color=[BLUE, RED], edgecolor="white", alpha=0.85)
    ax1.set_title("Transaction Class Distribution", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.spines[["top","right"]].set_visible(False)
    for i, (label, count) in enumerate(zip(labels, counts)):
        ax1.text(i, count + 100, f"{count:,}", ha="center", fontsize=9, fontweight="bold")

    # Panel 2: ML Model Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    if ml_results:
        models = list(ml_results.keys())
        aucs   = [ml_results[m]["auc"] for m in models]
        f1s    = [ml_results[m]["f1"]  for m in models]
        x = np.arange(len(models))
        w = 0.35
        ax2.bar(x - w/2, aucs, w, label="AUC",    color=BLUE,   edgecolor="white", alpha=0.85)
        ax2.bar(x + w/2, f1s,  w, label="F1",     color=GREEN,  edgecolor="white", alpha=0.85)
        ax2.set_xticks(x); ax2.set_xticklabels(models)
        ax2.set_ylim(0, 1); ax2.set_title("ML Model Comparison", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9); ax2.spines[["top","right"]].set_visible(False)
        for i, (auc, f1) in enumerate(zip(aucs, f1s)):
            ax2.text(i - w/2, auc + 0.01, f"{auc:.3f}", ha="center", fontsize=8)
            ax2.text(i + w/2, f1  + 0.01, f"{f1:.3f}",  ha="center", fontsize=8)

    # Panel 3: Graph Pattern Counts
    ax3 = fig.add_subplot(gs[0, 2])
    if graph_results:
        patterns = ["Fan-Out", "Fan-In", "Cycles"]
        counts_g = [
            graph_results["fan_out_count"],
            graph_results["fan_in_count"],
            graph_results["cycle_count"],
        ]
        colors_g = [RED if c > 10 else ORANGE if c > 5 else GREEN for c in counts_g]
        bars = ax3.bar(patterns, counts_g, color=colors_g, edgecolor="white", alpha=0.85)
        ax3.set_title(
            f"Graph Patterns Detected\n({graph_results['flagged_accounts']:,} flagged accounts)",
            fontsize=11, fontweight="bold"
        )
        ax3.set_ylabel("Pattern Count"); ax3.spines[["top","right"]].set_visible(False)
        for bar, count in zip(bars, counts_g):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     str(count), ha="center", fontsize=10, fontweight="bold")

    # Panel 4: Amount distribution by class
    ax4 = fig.add_subplot(gs[1, :2])
    if trans_pdf is not None and len(trans_pdf) > 0:
        legit = trans_pdf[trans_pdf["is_laundering"] == 0]["amount_paid"]
        laund = trans_pdf[trans_pdf["is_laundering"] == 1]["amount_paid"]
        legit_clipped = legit.clip(upper=legit.quantile(0.99))
        laund_clipped = laund.clip(upper=laund.quantile(0.99))
        ax4.hist(legit_clipped, bins=50, alpha=0.6, color=BLUE,  label="Legitimate", density=True)
        ax4.hist(laund_clipped, bins=50, alpha=0.6, color=RED,   label="Laundering", density=True)
        ax4.set_title("Transaction Amount Distribution by Class (99th pct clip)",
                      fontsize=11, fontweight="bold")
        ax4.set_xlabel("Amount Paid"); ax4.set_ylabel("Density")
        ax4.legend(fontsize=9); ax4.spines[["top","right"]].set_visible(False)

    # Panel 5: Precision / Recall
    ax5 = fig.add_subplot(gs[1, 2])
    if ml_results:
        models = list(ml_results.keys())
        prec   = [ml_results[m]["precision"] for m in models]
        rec    = [ml_results[m]["recall"]    for m in models]
        x = np.arange(len(models))
        w = 0.35
        ax5.bar(x - w/2, prec, w, label="Precision", color=ORANGE, edgecolor="white", alpha=0.85)
        ax5.bar(x + w/2, rec,  w, label="Recall",    color=GREEN,  edgecolor="white", alpha=0.85)
        ax5.set_xticks(x); ax5.set_xticklabels(models)
        ax5.set_ylim(0, 1)
        ax5.set_title("Precision vs Recall", fontsize=11, fontweight="bold")
        ax5.legend(fontsize=9); ax5.spines[["top","right"]].set_visible(False)

    # Panel 6: Summary stats
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis("off")
    best_model = max(ml_results, key=lambda m: ml_results[m]["auc"]) if ml_results else "N/A"
    best = ml_results.get(best_model, {})
    summary_text = (
        f"DATASET:  {summary['total_transactions']:,} transactions | "
        f"{summary['laundering_transactions']:,} laundering ({summary['laundering_pct']:.2f}%)\n\n"
        f"BEST MODEL ({best_model}):  "
        f"AUC={best.get('auc','N/A')} | F1={best.get('f1','N/A')} | "
        f"Precision={best.get('precision','N/A')} | Recall={best.get('recall','N/A')}\n\n"
        f"GRAPH ANALYSIS:  "
        f"{graph_results.get('nodes',0):,} accounts | "
        f"{graph_results.get('edges',0):,} transactions | "
        f"{graph_results.get('flagged_accounts',0):,} flagged accounts | "
        f"{graph_results.get('cycle_count',0)} cycles detected"
    )
    ax6.text(0.05, 0.85, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#E6EAF0", alpha=0.8))

    from datetime import datetime
    fig.text(0.5, 0.01,
             f"Data: IBM AML Synthetic Dataset | Generated: {datetime.now().strftime('%Y-%m-%d')} | rajapalagummi.com",
             ha="center", fontsize=8, color="#999", style="italic")

    path = os.path.join(OUTPUT_DIR, "aml_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Dashboard] → {path}")
    return path
