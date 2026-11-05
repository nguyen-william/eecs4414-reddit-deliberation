"""Gao et al. deliberation metrics and subreddit ranking (P10).

Implements:
  - max/avg depth and width of a reply tree (Gao / González-Bailón style)
  - Quadrant-I boundaries so ~10% of all posts are in Q1 (high depth AND high width)
  - P10 = fraction of a subreddit's posts in Quadrant I (their ranking score)
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

import numpy as np
import pandas as pd


def _children_map(ids: Iterable[str], parents: Iterable[str]) -> dict[str, list[str]]:
    id_set = set(ids)
    children: dict[str, list[str]] = defaultdict(list)
    for uid, parent in zip(ids, parents):
        if parent and parent in id_set and parent != uid:
            children[parent].append(uid)
    return children


def _roots(ids: list[str], parents: list[str], id_set: set[str]) -> list[str]:
    roots = []
    for uid, parent in zip(ids, parents):
        if (not parent) or (parent not in id_set) or (parent == uid):
            roots.append(uid)
    return roots or ([ids[0]] if ids else [])


def tree_depth_width_stats(utts: pd.DataFrame) -> dict:
    """Return max/avg depth and width for one thread (conversation).

    Depth of a node = #edges from root.
    Max depth = deepest node.
    Avg depth = mean depth of sink nodes (leaves), per Gao et al.
    Width at depth d = #nodes at that depth.
    Max width = max over depths.
    Avg width = mean width over depths that appear.
    """
    if utts.empty:
        return {
            "max_depth": 0,
            "avg_depth": 0.0,
            "max_width": 0,
            "avg_width": 0.0,
            "n_comments": 0,
        }

    ids = utts["utterance_id"].astype(str).tolist()
    parents = utts["reply_to"].fillna("").astype(str).tolist()
    id_set = set(ids)
    children = _children_map(ids, parents)
    roots = _roots(ids, parents, id_set)

    # BFS depths from all roots (forest)
    depth: dict[str, int] = {}
    q: deque[str] = deque()
    for r in roots:
        depth[r] = 0
        q.append(r)
    while q:
        node = q.popleft()
        for ch in children.get(node, []):
            if ch in depth:
                continue
            depth[ch] = depth[node] + 1
            q.append(ch)

    # any unreachable nodes: treat as depth 0 roots
    for uid in ids:
        if uid not in depth:
            depth[uid] = 0

    level_counts: dict[int, int] = defaultdict(int)
    for d in depth.values():
        level_counts[d] += 1

    sinks = [uid for uid in ids if not children.get(uid)]
    sink_depths = [depth[uid] for uid in sinks] or [0]

    widths = [level_counts[d] for d in sorted(level_counts)]
    return {
        "max_depth": int(max(depth.values()) if depth else 0),
        "avg_depth": float(np.mean(sink_depths)),
        "max_width": int(max(widths) if widths else 0),
        "avg_width": float(np.mean(widths) if widths else 0.0),
        "n_comments": int(len(ids)),
    }


def per_post_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """One row per conversation/post with Gao depth/width metrics."""
    rows = []
    for conv_id, g in df.groupby("conversation_id"):
        stats = tree_depth_width_stats(g)
        rows.append(
            {
                "conversation_id": conv_id,
                "subreddit": g["subreddit"].iloc[0],
                **stats,
            }
        )
    return pd.DataFrame(rows)


def find_q1_thresholds(
    avg_depth: np.ndarray,
    avg_width: np.ndarray,
    target_frac: float = 0.10,
) -> tuple[float, float, float]:
    """Find depth/width cutoffs so Quadrant I ≈ target_frac of posts.

    Q1 = avg_depth >= td AND avg_width >= tw.
    Searches percentile pairs and picks the closest to target_frac.
    """
    best = (0.0, 0.0, 1e9, 0.0)  # td, tw, abs_err, frac
    for pd_ in range(50, 96):
        for pw in range(50, 96):
            td = float(np.percentile(avg_depth, pd_))
            tw = float(np.percentile(avg_width, pw))
            frac = float(((avg_depth >= td) & (avg_width >= tw)).mean())
            err = abs(frac - target_frac)
            # prefer closer frac; tie-break toward slightly <= target
            if err < best[2] or (np.isclose(err, best[2]) and frac <= target_frac and best[3] > target_frac):
                best = (td, tw, err, frac)
    return best[0], best[1], best[3]


def assign_quadrant(avg_depth: float, avg_width: float, td: float, tw: float) -> str:
    high_d = avg_depth >= td
    high_w = avg_width >= tw
    if high_d and high_w:
        return "I"
    if high_d and not high_w:
        return "II"
    if (not high_d) and (not high_w):
        return "III"
    return "IV"


def rank_subreddits_gao(
    post_df: pd.DataFrame,
    target_frac: float = 0.10,
    use_avg: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Rank subreddits by Gao P10 (share of posts in Quadrant I).

    Returns (ranking_table, threshold_info).
    """
    if use_avg:
        xs = post_df["avg_width"].to_numpy(dtype=float)
        ys = post_df["avg_depth"].to_numpy(dtype=float)
        metric_name = "P10_avg"
    else:
        xs = post_df["max_width"].to_numpy(dtype=float)
        ys = post_df["max_depth"].to_numpy(dtype=float)
        metric_name = "P10_max"

    # find_q1_thresholds(avg_depth, avg_width) → (td, tw, realized_frac)
    td, tw, realized = find_q1_thresholds(ys, xs, target_frac=target_frac)

    quads = [
        assign_quadrant(float(y), float(x), td, tw) for y, x in zip(ys, xs)
    ]
    out = post_df.copy()
    out["quadrant"] = quads
    out["in_q1"] = out["quadrant"] == "I"

    ranking = (
        out.groupby("subreddit")
        .agg(
            n_posts=("conversation_id", "count"),
            n_q1=("in_q1", "sum"),
            mean_avg_depth=("avg_depth", "mean"),
            mean_avg_width=("avg_width", "mean"),
            mean_max_depth=("max_depth", "mean"),
            mean_max_width=("max_width", "mean"),
        )
        .reset_index()
    )
    ranking[metric_name] = ranking["n_q1"] / ranking["n_posts"]
    ranking = ranking.sort_values(metric_name, ascending=False)
    ranking["rank"] = range(1, len(ranking) + 1)

    info = {
        "metric": metric_name,
        "target_frac": target_frac,
        "realized_global_q1_frac": realized,
        "depth_threshold": td,
        "width_threshold": tw,
        "definition": (
            "Quadrant I = depth >= td AND width >= tw, with (td, tw) chosen so "
            f"about {target_frac:.0%} of all posts fall in Q1 (Gao et al.). "
            "Subreddit score = share of its posts in Q1."
        ),
    }
    return ranking, info


def rank_on_utterances(df: pd.DataFrame, target_frac: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    posts = per_post_metrics(df)
    ranking, info = rank_subreddits_gao(posts, target_frac=target_frac, use_avg=True)
    return posts, ranking, info
