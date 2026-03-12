import networkx as nx
import pandas as pd


def calculate_pagerank(df: pd.DataFrame) -> dict:
    """
    Calculate PageRank for all articles based on citation network.

    PageRank measures importance based on WHO cites you, not just how many.
    Being cited by high-PageRank articles gives you higher PageRank.

    Returns:
        Dictionary mapping PMID (str) -> PageRank score
    """
    print("Building citation graph for PageRank...")

    # Build directed graph: edge A→B means "A cites B"
    G = nx.DiGraph()

    # Add all articles as nodes
    pmid_set = set(df["Source_PMID"].astype(str))
    G.add_nodes_from(pmid_set)

    # Add edges from Cites_PMIDs (outgoing citations)
    edge_count = 0
    for _, row in df.iterrows():
        source = str(row["Source_PMID"])
        if pd.notna(row["Cites_PMIDs"]) and row["Cites_PMIDs"]:
            for cited in str(row["Cites_PMIDs"]).split(";"):
                cited = cited.strip()
                if cited in pmid_set:  # Only internal edges
                    G.add_edge(source, cited)
                    edge_count += 1

    print(f"  Graph: {G.number_of_nodes():,} nodes, {edge_count:,} internal edges")

    # Run PageRank with damping factor 0.85
    print("  Running PageRank algorithm (alpha=0.85)...")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)

    return pagerank