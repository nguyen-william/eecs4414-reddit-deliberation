import networkx as nx
import pandas as pd

from src.actor_layer import build_actor_layer, basic_stats
from src.network import null_models, avg_clustering, analysis_settings, degree_ccdf, expected_er_clustering, small_world_sigma

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


def test_sampled_clustering_close_to_exact():
    U = nx.powerlaw_cluster_graph(3000, 4, 0.3, seed=1)
    exact = avg_clustering(U)
    sampled = avg_clustering(U, n_nodes=1500, seed=0)
    assert abs(exact - sampled) < 0.03


def test_settings_switch_for_big_graphs():
    assert analysis_settings(25_000)["clustering_nodes"] is None
    assert analysis_settings(500_000)["clustering_nodes"] == 20_000


def test_degree_ccdf():
    G, _ = build_actor_layer(toy_comments(), POST_AUTHORS)
    x, ccdf = degree_ccdf(G)
    assert list(x) == [1, 2]
    assert list(ccdf) == [1.0, 0.5]


def test_expected_er_clustering_matches_random_graphs():
    n, m = 2000, 10000
    measured = nx.average_clustering(nx.gnm_random_graph(n, m, seed=0))
    assert abs(measured - expected_er_clustering(n, m)) < 0.002


def test_sigma_is_one_for_a_random_graph_itself():
    n, m = 1000, 5000
    p = expected_er_clustering(n, m)
    assert abs(small_world_sigma(p, 3.0, n, m, 3.0) - 1.0) < 1e-9


if __name__ == "__main__":
    tests = [f for name, f in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"passed: {test.__name__}")
    print(f"all {len(tests)} tests passed")
