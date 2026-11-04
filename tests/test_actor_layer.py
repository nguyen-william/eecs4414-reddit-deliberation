import networkx as nx
import pandas as pd

from src.actor_layer import build_actor_layer, basic_stats
from src.network import null_models

POST_AUTHORS = {"T1": "Mona"}


def toy_comments():
    return pd.DataFrame({
        "comment_id": ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"],
        "thread_id":  ["T1"] * 8,
        "parent_id":  ["T1", "c1", "c2", "T1", "c4", "c99", "c1", "c3"],
        "author":     ["Ali", "Sara", "Ali", "Ben", "Ben", "Sara", "AutoModerator", "Sara"],
    })


def test_edges_and_weights():
    G, _ = build_actor_layer(toy_comments(), POST_AUTHORS)
    edges = {(u, v): w for u, v, w in G.edges(data="weight")}
    assert edges == {("Ali", "Mona"): 1, ("Sara", "Ali"): 2, ("Ali", "Sara"): 1, ("Ben", "Mona"): 1}


def test_skipped_replies():
    _, skipped = build_actor_layer(toy_comments(), POST_AUTHORS)
    assert skipped["self_reply"] == 1
    assert skipped["parent_missing"] == 1


def test_bots_removed():
    G, _ = build_actor_layer(toy_comments(), POST_AUTHORS)
    assert "AutoModerator" not in G


def test_without_post_authors():
    G, skipped = build_actor_layer(toy_comments())
    assert "Mona" not in G
    assert skipped["post_author_unknown"] == 2


def test_basic_stats():
    G, _ = build_actor_layer(toy_comments(), POST_AUTHORS)
    stats = basic_stats(G)
    assert stats["nodes"] == 4
    assert stats["edges"] == 4
    assert stats["total_replies"] == 5
    assert stats["reciprocity"] == 0.5


def test_null_models_report_spread():
    U = nx.karate_club_graph()
    nulls = null_models(U, n_samples=3)
    for key in ["C_er", "L_er", "C_config", "L_config"]:
        assert nulls[key] > 0
        assert nulls[key + "_std"] >= 0


if __name__ == "__main__":
    tests = [f for name, f in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"passed: {test.__name__}")
    print(f"all {len(tests)} tests passed")
