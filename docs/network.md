# Actor layer + network analysis (RQ1) — Omid

Builds the user network (actor layer) for each subreddit and measures it.

## Data
Download the shared Drive folder into `data/` at the repo root (git ignores it). The code expects Beyza's names:
`<sub>_2017_final.csv` (comments) and, when available, `<sub>_2017_post_authors_final.csv` next to it.

## Run
```bash
pip install -r requirements.txt
python3 -m scripts.run_network --data data                    # every <sub>_2017_final.csv in data/
python3 -m scripts.run_network canada=data/canada_2017_final.csv   # or one/several subreddits by name
python3 -m tests.test_actor_layer                             # unit tests
```
Each subreddit's results are saved on their own (`results/rq1_rows/`, `results/degree_ccdf/`), and the table and
figure are rebuilt from everything saved so far. You can run subreddits one at a time, in any order.

## Files
- `src/actor_layer.py` — builds the weighted directed user network: edge A -> B = A replied to B (weight = count).
  Replies to the post go to the post author. Bots, [deleted] users and self-replies are skipped.
- `src/network.py` — clustering, average path length, ER + configuration-model comparisons (mean and std),
  small-world sigma, power-law fit (vs lognormal), degree / PageRank / betweenness rankings, degree CCDF.
- `scripts/run_network.py` — runs everything and saves the outputs below.
- `scripts/run_actor_layer.py` — quick check that prints basic counts.
- `tests/test_actor_layer.py` — unit tests on a hand-made thread with a known answer.

## Big graphs (more than 50,000 users in the largest component)
Exact measures are too slow there, so `analysis_settings()` switches to sampling (same seed every run):
clustering on 20,000 random users, path length from 200 random users (50 for each random graph),
3 random graphs of each kind instead of 5, betweenness estimated from 100 sampled users instead of 500.
The settings used are saved in the `setting_*` columns of `results/rq1_network.csv`.

## Outputs
- `results/rq1_network.csv` — one row per subreddit: data counts, skipped replies, network properties (Table 3).
- `results/fig2_degree_ccdf.pdf` — degree distribution, log-log, all subreddits (Figure 2).
- `results/baselines/<sub>_<method>.csv` — top 10% of users ranked by degree, PageRank, betweenness
  (`user, score`, highest first), used as baselines for RQ2 (k goes up to 10% of users).
- `results/rq1_rows/<sub>.json`, `results/degree_ccdf/<sub>.csv` — per-subreddit pieces the table and figure are built from.

## Small-world sigma
sigma = (C / C_ER) / (L / L_ER), with C_ER = 2m / (n(n-1)), the exact expected clustering of a G(n,m) random graph
(column `C_er_expected`). The sampled ER clustering (`C_er`) is still reported, but for huge sparse graphs it can be
almost 0 from sampling noise, so it is not used for sigma.
