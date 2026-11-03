from collections import Counter

import networkx as nx
import pandas as pd

KNOWN_BOTS = {"AutoModerator", "WikiTextBot", "_Coffeebot", "sneakpeekbot", "youtubefactsbot",
              "HelperBot_", "imguralbumbot", "GoodBot_BadBot"}


def load_comments(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"comment_id": str, "thread_id": str, "parent_id": str, "author": str})
    df = df.dropna(subset=["comment_id", "parent_id", "author"])
    df = df[df["author"] != "[deleted]"]
    return df


def load_post_authors(path: str) -> dict:
    posts = pd.read_csv(path, dtype=str).dropna()
    posts = posts[(posts["post_author"] != "[deleted]") & (~posts["post_author"].isin(KNOWN_BOTS))]
    return dict(zip(posts["thread_id"], posts["post_author"]))


def build_actor_layer(comments: pd.DataFrame, post_authors: dict = None, drop_bots: bool = True):
    df = comments[~comments["author"].isin(KNOWN_BOTS)] if drop_bots else comments
    author_of = dict(zip(df["comment_id"], df["author"]))
    post_authors = post_authors or {}

    weights = Counter()
    skipped = Counter()
    for parent_id, thread_id, author in zip(df["parent_id"], df["thread_id"], df["author"]):
        if parent_id == thread_id:
            target = post_authors.get(thread_id)
        else:
            target = author_of.get(parent_id)
        if target is None:
            skipped["post_author_unknown" if parent_id == thread_id else "parent_missing"] += 1
            continue
        if target == author:
            skipped["self_reply"] += 1
            continue
        weights[(author, target)] += 1

    G = nx.DiGraph()
    G.add_nodes_from(df["author"].unique())
    G.add_nodes_from(set(post_authors.get(t) for t in df["thread_id"].unique()) - {None})
    G.add_weighted_edges_from((u, v, w) for (u, v), w in weights.items())
    return G, skipped


def basic_stats(G: nx.DiGraph) -> dict:
    U = G.to_undirected()
    lcc = max(nx.connected_components(U), key=len)
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "total_replies": int(sum(w for _, _, w in G.edges(data="weight"))),
        "isolated_users": nx.number_of_isolates(G),
        "mean_degree": round(2 * U.number_of_edges() / U.number_of_nodes(), 2),
        "density": round(nx.density(G), 6),
        "reciprocity": round(nx.reciprocity(G), 3),
        "lcc_share": round(len(lcc) / U.number_of_nodes(), 3),
    }
