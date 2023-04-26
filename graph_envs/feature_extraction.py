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
    eigen = nx.eigenvector_centrality(G)
    
    sf = []
    
    for node in G.nodes:
        # # Add node degrees:
        # G.nodes[node]['degree'] = G.degree(node)
        
        # # Add centrality measures:
        # G.nodes[node]['betweenness'] = betweenness[node]
        # G.nodes[node]['closeness'] = closeness[node]
        # G.nodes[node]['eigen'] = eigen[node]
        
        # # Add PageRank:
        # G.nodes[node]['pr'] = pr[node]

        # # Add clustering coefficients:
        # G.nodes[node]['clustering'] = clustering[node]
        
        # sf.append([G.nodes[node]['degree'], G.nodes[node]['betweenness'], G.nodes[node]['closeness'], G.nodes[node]['eigen'], G.nodes[node]['pr'], G.nodes[node]['clustering']])
        sf.append([G.degree(node), betweenness[node], closeness[node], eigen[node], pr[node], clustering[node]])
    
    sf = torch.tensor(sf)
    return sf


def get_num_features():
    return 6