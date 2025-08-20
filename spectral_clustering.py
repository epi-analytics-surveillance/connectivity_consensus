import scipy.stats
from sklearn.cluster import SpectralClustering


# Total number of data points
n = 3

# Total umber of clusters at each MCMC iteration
num_clusters_chain = [[2, 2, 3],]

# MCMC samples of cluster assignments
# [1, 1, 2] = an MCMC sample where data point 1 is in cluster 1, data point 2 is in cluster 1, and data point 3 is in cluster 2
assign_chain = [[[1, 1, 2], [1, 1, 2], [1, 2, 3],],]


def compute_connectivity_matrix(assign_chain, n):
    """Compute an n x n matrix, where entry (i, j) is the number of times data point i and data point j were in the same cluster.
    """
    connectivity = np.zeros((n, n))

    for model_assignments in assign_chain:
        for assignments in model_assignments:
            for i, cluster_i in enumerate(assignments):
                for j, cluster_j in enumerate(assignments):
                    if i != j and cluster_i == cluster_j:
                        connectivity[i, j] += 1
                    if i == j:
                        connectivity[i, j] += 1
    return connectivity

    
def consensus_clustering(num_clusters_chain, assign_chain, n):
    num_clusters_samples = [x for sublist in num_clusters_chain for x in sublist]
    num_clusters = scipy.stats.mode(num_clusters_samples)[0]
    
    connectivity = compute_connectivity_matrix(assign_chain, n)
    
    m = SpectralClustering(n_clusters=num_clusters, affinity='precomputed').fit(connectivity)
    
    return num_clusters, m.labels_


clustering = consensus_clustering(num_clusters_chain, assign_chain, n)[1]
