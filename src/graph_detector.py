import networkx as nx
import pandas as pd
from collections import defaultdict


def build_graph(transactions_pdf):
    G = nx.DiGraph()
    for _, row in transactions_pdf.iterrows():
        G.add_edge(
            row["from_account"],
            row["to_account"],
            amount=row["amount_paid"],
            currency=row["payment_currency"],
            is_laundering=row["is_laundering"],
        )
    return G


def detect_fan_out(G, threshold=5):
    flagged = []
    for node in G.nodes():
        out_degree = G.out_degree(node)
        if out_degree >= threshold:
            neighbors = list(G.successors(node))
            laundering_edges = sum(
                1 for n in neighbors
                if G[node][n].get("is_laundering", 0) == 1
            )
            flagged.append({
                "account": node,
                "pattern": "FAN_OUT",
                "degree": out_degree,
                "laundering_edges": laundering_edges,
            })
    return flagged


def detect_fan_in(G, threshold=5):
    flagged = []
    for node in G.nodes():
        in_degree = G.in_degree(node)
        if in_degree >= threshold:
            predecessors = list(G.predecessors(node))
            laundering_edges = sum(
                1 for p in predecessors
                if G[p][node].get("is_laundering", 0) == 1
            )
            flagged.append({
                "account": node,
                "pattern": "FAN_IN",
                "degree": in_degree,
                "laundering_edges": laundering_edges,
            })
    return flagged


def detect_cycles(G, max_length=10):
    flagged = []
    try:
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            if 2 <= len(cycle) <= max_length:
                laundering = sum(
                    1 for i in range(len(cycle))
                    if G[cycle[i]][cycle[(i + 1) % len(cycle)]].get("is_laundering", 0) == 1
                )
                flagged.append({
                    "accounts": cycle,
                    "pattern": "CYCLE",
                    "length": len(cycle),
                    "laundering_edges": laundering,
                })
    except Exception:
        pass
    return flagged


def run_graph_analysis(transactions_pdf):
    print("[GraphDetector] Building transaction graph...")
    G = build_graph(transactions_pdf)

    print(f"[GraphDetector] Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,}")

    fan_out = detect_fan_out(G, threshold=5)
    fan_in  = detect_fan_in(G, threshold=5)
    cycles  = detect_cycles(G, max_length=10)

    total_flagged = len(set(
        [f["account"] for f in fan_out] +
        [f["account"] for f in fan_in] +
        [a for c in cycles for a in c["accounts"]]
    ))

    print(f"[GraphDetector] Fan-Out patterns:  {len(fan_out)}")
    print(f"[GraphDetector] Fan-In patterns:   {len(fan_in)}")
    print(f"[GraphDetector] Cycle patterns:    {len(cycles)}")
    print(f"[GraphDetector] Flagged accounts:  {total_flagged:,}")

    return {
        "nodes":           G.number_of_nodes(),
        "edges":           G.number_of_edges(),
        "fan_out_count":   len(fan_out),
        "fan_in_count":    len(fan_in),
        "cycle_count":     len(cycles),
        "flagged_accounts":total_flagged,
        "fan_out":         fan_out[:20],
        "fan_in":          fan_in[:20],
        "cycles":          cycles[:20],
    }


if __name__ == "__main__":
    from src.data_pipeline import get_spark, load_transactions

    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")

    df = load_transactions(spark, "data/HI-Small_Trans.csv")
    pdf = df.select(
        "from_account", "to_account",
        "amount_paid", "payment_currency", "is_laundering"
    ).toPandas()

    results = run_graph_analysis(pdf)
    spark.stop()
