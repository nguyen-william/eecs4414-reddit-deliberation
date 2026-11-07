import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import powerlaw

BIG_GRAPH = 50_000


def sig(x: float, digits: int = 4) -> float:
    return float(f"{float(x):.{digits}g}")


def expected_er_clustering(n: int, m: int) -> float:
    return 2 * m / (n * (n - 1))


def small_world_sigma(C: float, L: float, n: int, m: int, L_er: float) -> float:
    return (C / expected_er_clustering(n, m)) / (L / L_er)


def analysis_settings(n_nodes: int) -> dict:
    if n_nodes <= BIG_GRAPH:
        return {"clustering_nodes": None, "path_sources": 500, "null_samples": 5,
                "null_path_sources": 200, "betweenness_k": 500}
    return {"clustering_nodes": 20_000, "path_sources": 200, "null_samples": 3,
            "null_path_sources": 50, "betweenness_k": 100}


def simple_undirected(G: nx.DiGraph) -> nx.Graph:
    U = nx.Graph(G.to_undirected())
    U.remove_edges_from(nx.selfloop_edges(U))
    return U


def largest_component(U: nx.Graph) -> nx.Graph:
    nodes = max(nx.connected_components(U), key=len)
    return U.subgraph(nodes).copy()


def avg_clustering(U: nx.Graph, n_nodes: int = None, seed: int = 0) -> float:
    if n_nodes is None or U.number_of_nodes() <= n_nodes:
        return nx.average_clustering(U)
    nodes = random.Random(seed).sample(list(U.nodes()), n_nodes)
    return float(np.mean(list(nx.clustering(U, nodes=nodes).values())))


def avg_path_length(U: nx.Graph, n_sources: int = 500, seed: int = 0) -> float:
    rng = random.Random(seed)
    nodes = list(U.nodes())
    sources = nodes if len(nodes) <= n_sources else rng.sample(nodes, n_sources)
    total, count = 0, 0
    for s in sources:
        lengths = nx.single_source_shortest_path_length(U, s)
        total += sum(lengths.values())
        count += len(lengths) - 1
    return total / count


def power_law_fit(U: nx.Graph) -> dict:
    degrees = [d for _, d in U.degree() if d > 0]
    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
    R, p = fit.distribution_compare("power_law", "lognormal")
    return {"alpha": round(float(fit.power_law.alpha), 2), "xmin": int(fit.power_law.xmin),
            "pl_vs_lognormal_R": round(float(R), 2), "pl_vs_lognormal_p": round(float(p), 3)}


def null_models(U: nx.Graph, n_samples: int = 5, seed: int = 0, n_sources: int = 200,
                clustering_nodes: int = None) -> dict:
    n, m = U.number_of_nodes(), U.number_of_edges()
    degree_seq = [d for _, d in U.degree()]
    results = {"C_er": [], "L_er": [], "C_config": [], "L_config": []}
    for i in range(n_samples):
        er = largest_component(nx.gnm_random_graph(n, m, seed=seed + i))
        results["C_er"].append(avg_clustering(er, clustering_nodes, seed + i))
        results["L_er"].append(avg_path_length(er, n_sources, seed + i))
        del er

        cm = nx.Graph(nx.configuration_model(degree_seq, seed=seed + i))
        cm.remove_edges_from(nx.selfloop_edges(cm))
        cm = largest_component(cm)
        results["C_config"].append(avg_clustering(cm, clustering_nodes, seed + i))
        results["L_config"].append(avg_path_length(cm, n_sources, seed + i))
        del cm
    summary = {}
    for k, v in results.items():
        summary[k] = sig(np.mean(v))
        summary[k + "_std"] = sig(np.std(v))
    return summary


def network_properties(G: nx.DiGraph, seed: int = 0) -> dict:
    U = largest_component(simple_undirected(G))
    s = analysis_settings(U.number_of_nodes())
    C = avg_clustering(U, s["clustering_nodes"], seed)
    L = avg_path_length(U, s["path_sources"], seed)
    nulls = null_models(U, n_samples=s["null_samples"], seed=seed, n_sources=s["null_path_sources"],
                        clustering_nodes=s["clustering_nodes"])
    n, m = U.number_of_nodes(), U.number_of_edges()
    sigma = small_world_sigma(C, L, n, m, nulls["L_er"])
    row = {"lcc_nodes": n, "lcc_edges": m, "C": sig(C), "L": sig(L), **nulls,
           "C_er_expected": sig(expected_er_clustering(n, m)), "sigma": round(sigma, 1)}
    row.update(power_law_fit(U))
    row.update({f"setting_{k}": ("all" if v is None else v) for k, v in s.items()})
    return row


def centrality_rankings(G: nx.DiGraph, betweenness_k: int = None, seed: int = 0) -> dict:
    U = simple_undirected(G)
    if betweenness_k is None:
        betweenness_k = analysis_settings(U.number_of_nodes())["betweenness_k"]
    return {
        "degree": dict(U.degree()),
        "pagerank": nx.pagerank(G, weight="weight"),
        "betweenness": nx.betweenness_centrality(U, k=min(betweenness_k, U.number_of_nodes()), seed=seed),
    }


def degree_ccdf(G: nx.DiGraph) -> tuple:
    degrees = np.sort(np.array([d for _, d in simple_undirected(G).degree() if d > 0]))
    x = np.unique(degrees)
    ccdf = 1.0 - np.searchsorted(degrees, x, side="left") / len(degrees)
    return x, ccdf


def plot_degree_ccdf(curves: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for name, (x, ccdf) in curves.items():
        ax.loglog(x, ccdf, marker=".", linestyle="none", markersize=3, label=name)
    ax.set_xlabel("Degree k (number of users interacted with)")
    ax.set_ylabel("P(degree ≥ k)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
