import glob
import json
import os
import sys
import time

import pandas as pd

from src.actor_layer import load_comments, load_post_authors, build_actor_layer, basic_stats
from src.network import network_properties, centrality_rankings, degree_ccdf, plot_degree_ccdf

ROWS_DIR = "results/rq1_rows"
BASELINE_TOP_SHARE = 0.10
CCDF_DIR = "results/degree_ccdf"


def find_inputs(args: list) -> list:
    if args and args[0] == "--data":
        paths = sorted(glob.glob(os.path.join(args[1], "*_2017_final.csv")))
        return [(os.path.basename(p).split("_")[0], p) for p in paths]
    return [tuple(arg.split("=")) for arg in args]


def run_one(name: str, path: str) -> None:
    start = time.time()
    posts_path = path.replace("_final.csv", "_post_authors_final.csv")
    post_authors = load_post_authors(posts_path) if os.path.exists(posts_path) else None
    print(f"{name}: post authors {'found' if post_authors else 'MISSING, replies to posts skipped'}", flush=True)

    comments = load_comments(path)
    G, skipped = build_actor_layer(comments, post_authors)
    row = {"subreddit": name, "comments": len(comments), "threads": comments["thread_id"].nunique(),
           "post_authors_used": post_authors is not None,
           **{f"skipped_{k}": v for k, v in skipped.items()}}
    del comments

    row.update(basic_stats(G))
    row.update(network_properties(G))
    with open(f"{ROWS_DIR}/{name}.json", "w") as f:
        json.dump(row, f, indent=1)
    print(row, flush=True)

    x, ccdf = degree_ccdf(G)
    pd.DataFrame({"degree": x, "ccdf": ccdf}).to_csv(f"{CCDF_DIR}/{name}.csv", index=False)

    for method, scores in centrality_rankings(G).items():
        table = pd.DataFrame(scores.items(), columns=["user", "score"]).sort_values("score", ascending=False)
        top = table.head(max(1, round(len(table) * BASELINE_TOP_SHARE)))
        top.to_csv(f"results/baselines/{name}_{method}.csv", index=False)
    print(f"{name} done in {time.time() - start:.0f} s", flush=True)


def rebuild_outputs() -> None:
    rows = [json.load(open(p)) for p in sorted(glob.glob(f"{ROWS_DIR}/*.json"))]
    pd.DataFrame(rows).to_csv("results/rq1_network.csv", index=False)
    curves = {}
    for p in sorted(glob.glob(f"{CCDF_DIR}/*.csv")):
        df = pd.read_csv(p)
        curves[os.path.basename(p)[:-4]] = (df["degree"].to_numpy(), df["ccdf"].to_numpy())
    plot_degree_ccdf(curves, "results/fig2_degree_ccdf.pdf")
    print(f"saved results/rq1_network.csv ({len(rows)} subreddits) and results/fig2_degree_ccdf.pdf")


if __name__ == "__main__":
    for folder in [ROWS_DIR, CCDF_DIR, "results/baselines"]:
        os.makedirs(folder, exist_ok=True)
    for name, path in find_inputs(sys.argv[1:]):
        run_one(name, path)
    rebuild_outputs()
