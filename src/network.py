import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import powerlaw


def simple_undirected(G: nx.DiGraph) -> nx.Graph:
    U = nx.Graph(G.to_undirected())
    U.remove_edges_from(nx.selfloop_edges(U))
    return U


def largest_component(U: nx.Graph) -> nx.Graph:
    nodes = max(nx.connected_components(U), key=len)
    return U.subgraph(nodes).copy()


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


def null_models(U: nx.Graph, n_samples: int = 5, seed: int = 0, n_sources: int = 200) -> dict:
    n, m = U.number_of_nodes(), U.number_of_edges()
    degree_seq = [d for _, d in U.degree()]
    results = {"C_er": [], "L_er": [], "C_config": [], "L_config": []}
    for i in range(n_samples):
        er = largest_component(nx.gnm_random_graph(n, m, seed=seed + i))
        results["C_er"].append(nx.average_clustering(er))
        results["L_er"].append(avg_path_length(er, n_sources, seed + i))

        cm = nx.Graph(nx.configuration_model(degree_seq, seed=seed + i))
        cm.remove_edges_from(nx.selfloop_edges(cm))
        cm = largest_component(cm)
        results["C_config"].append(nx.average_clustering(cm))
        results["L_config"].append(avg_path_length(cm, n_sources, seed + i))
    summary = {}
    for k, v in results.items():
        summary[k] = round(float(np.mean(v)), 4)
        summary[k + "_std"] = round(float(np.std(v)), 4)
    return summary


def network_properties(G: nx.DiGraph, n_samples: int = 5, seed: int = 0) -> dict:
    U = largest_component(simple_undirected(G))
    C = nx.average_clustering(U)
    L = avg_path_length(U, seed=seed)
    nulls = null_models(U, n_samples=n_samples, seed=seed)
    sigma = (C / nulls["C_er"]) / (L / nulls["L_er"])
    row = {"lcc_nodes": U.number_of_nodes(), "lcc_edges": U.number_of_edges(),
           "C": round(C, 4), "L": round(L, 3), **nulls, "sigma": round(sigma, 1)}
    row.update(power_law_fit(U))
    return row


def centrality_rankings(G: nx.DiGraph, betweenness_k: int = 500, seed: int = 0) -> dict:
    U = simple_undirected(G)
    return {
        "degree": dict(U.degree()),
        "pagerank": nx.pagerank(G, weight="weight"),
        "betweenness": nx.betweenness_centrality(U, k=min(betweenness_k, U.number_of_nodes()), seed=seed),
    }


def plot_degree_ccdf(graphs: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for name, G in graphs.items():
        degrees = np.array([d for _, d in simple_undirected(G).degree() if d > 0])
        x = np.sort(np.unique(degrees))
        ccdf = [(degrees >= k).mean() for k in x]
        ax.loglog(x, ccdf, marker=".", linestyle="none", markersize=3, label=name)
    ax.set_xlabel("Degree k (number of users interacted with)")
    ax.set_ylabel("P(degree ≥ k)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
