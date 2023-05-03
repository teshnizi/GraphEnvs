import networkx as nx
import torch



def generate_features(G: nx.Graph):
    
    """
    Add features to nodes in a graph.
    
    Parameters
    ----------
    G : nx.Graph
    
    Returns
    -------
    sf: A Tensor of shape (n_nodes, n_features) containing the structural features for each node.
    """
    
    # Add clustering coefficients:
    clustering = nx.clustering(G)

    # PageRank:
    pr = nx.pagerank(G)

    # Centrality:
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    # eigen = nx.eigenvector_centrality(G) THIS ONE CAUSES RECURSION ERRORS!
    
    sf = []
    
    for node in G.nodes:
        sf.append([G.degree(node), betweenness[node], closeness[node], pr[node], clustering[node]])
    
    sf = torch.tensor(sf)
    return sf


def get_num_features():
    return 5