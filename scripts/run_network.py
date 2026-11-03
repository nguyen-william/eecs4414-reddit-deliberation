import os
import sys
import time

import pandas as pd

from src.actor_layer import load_comments, load_post_authors, build_actor_layer, basic_stats
from src.network import network_properties, centrality_rankings, plot_degree_ccdf

os.makedirs("results/baselines", exist_ok=True)

graphs, rows = {}, []
for arg in sys.argv[1:]:
    name, path = arg.split("=")
    start = time.time()
    posts_path = path.replace("_final.csv", "_post_authors_final.csv")
    post_authors = load_post_authors(posts_path) if os.path.exists(posts_path) else None
    print(f"{name}: post authors {'found' if post_authors else 'MISSING, replies to posts skipped'}")
    G, skipped = build_actor_layer(load_comments(path), post_authors)
    print(f"{name}: skipped replies {dict(skipped)}")
    graphs[name] = G

    row = {"subreddit": name, **basic_stats(G), **network_properties(G)}
    rows.append(row)
    print(row)

    for method, scores in centrality_rankings(G).items():
        table = pd.DataFrame(scores.items(), columns=["user", "score"])
        table.sort_values("score", ascending=False).to_csv(f"results/baselines/{name}_{method}.csv", index=False)
    print(f"{name} done in {time.time() - start:.0f} s")

pd.DataFrame(rows).to_csv("results/rq1_network.csv", index=False)
plot_degree_ccdf(graphs, "results/fig2_degree_ccdf.pdf")
print("saved results/rq1_network.csv, results/fig2_degree_ccdf.pdf and results/baselines/")
