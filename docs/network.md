# Actor layer + network analysis (RQ1) — Omid

Builds the user network (actor layer) for each subreddit and measures it.

## Run
Put the data files from the shared Drive in one folder, e.g. `EECS4414/`, keeping Beyza's names
(`<sub>_2017_final.csv` and `<sub>_2017_post_authors_final.csv` side by side), then from the repo root:

```bash
pip install -r requirements.txt
python3 -m scripts.run_network toronto=EECS4414/toronto_2017_final.csv canada=EECS4414/canada_2017_final.csv
```

## Files
- `src/actor_layer.py` — builds the weighted directed user network: edge A -> B = A replied to B (weight = count).
  Replies to the post go to the post author. Bots, [deleted] users and self-replies are skipped.
- `src/network.py` — clustering, average path length, ER + configuration-model comparisons, small-world sigma,
  power-law fit (vs lognormal), and degree / PageRank / betweenness rankings.
- `scripts/run_network.py` — runs everything and saves the outputs below.
- `scripts/run_actor_layer.py` — quick check that prints basic counts.

## Outputs
- `results/rq1_network.csv` — network properties per subreddit (Table 3).
- `results/fig2_degree_ccdf.pdf` — degree distribution, log-log (Figure 2).
- `results/baselines/<sub>_<method>.csv` — users ranked by degree, PageRank, betweenness (`user, score`, highest first),
  used as baselines for RQ2.
