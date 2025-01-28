from rouge_score import rouge_scorer
from itertools import combinations

def calculate_rouge1_f1(sent1, sent2):
    """
    Compute the ROUGE-1 F1 score between two sentences.
    :param sent1: The first sentence (string)
    :param sent2: The second sentence (string)
    :return: ROUGE-1 F1 score (float)
    """
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(sent1, sent2)
    return scores['rouge1'].fmeasure  # Extract the F1 score

def calculate_sentence_distance(sent1, sent2):
    """
    Compute the distance between two sentences as dist(C1, C2) = 1 - R(C1, C2).
    :param sent1: The first sentence (string)
    :param sent2: The second sentence (string)
    :return: Distance between sentences (float)
    """
    rouge1_f1 = calculate_rouge1_f1(sent1, sent2)
    return 1 - rouge1_f1

def maximum_distance(cluster):
    """
    Compute the maximum distance between any two sentences of a cluster.
    :param cluster: The sentence cluster (list of sentences)
    :return: the maximum distance of any two sentences of a cluster (float)
    """
    max(calculate_sentence_distance(sent1, sent2) for sent1, sent2 in combinations(cluster, 2))

def average_distance(cluster):
    """
    Compute the average distance of all sentence pairs of a cluster.
    :param cluster: The sentence cluster (list of sentences)
    :return: the average distance of all sentence pairs of a cluster (float)
    """
    total_distance = 0
    pair_count = 0

    for sent1, sent2 in combinations(cluster, 2):
        total_distance += calculate_sentence_distance(sent1, sent2)
        pair_count += 1
    
    return total_distance / pair_count

def hausdorff_distance(cluster1, cluster2):
    """
    Compute the Hausdorff distance between two sentence clusters.
    :param cluster1: The first cluster (list of sentences)
    :param cluster2: The second cluster (list of sentences)
    :return: Hausdorff distance between the clusters (float)
    """
    # Calculate the infimum distances from cluster1 to cluster2
    inf_dist1 = max(
        min(calculate_sentence_distance(sent1, sent2) for sent2 in cluster2)
        for sent1 in cluster1
    )
    
    # Calculate the infimum distances from cluster2 to cluster1
    inf_dist2 = max(
        min(calculate_sentence_distance(sent2, sent1) for sent1 in cluster1)
        for sent2 in cluster2
    )
    
    # The Hausdorff distance is the maximum of the two distances
    return max(inf_dist1, inf_dist2)

def average_hausdorff_distance(clusters):
    """
    Compute the average Hausdorff distance between all pairs of sentence clusters.
    :param clusters: List of sentence clusters (list of lists of sentences)
    :return: Average Hausdorff distance (float)
    """
    n = len(clusters)  # Number of clusters
    total_distance = 0
    pair_count = 0

    # Iterate over all unique pairs of clusters
    for i in range(n):
        for j in range(i + 1, n):
            total_distance += hausdorff_distance(clusters[i], clusters[j])
            pair_count += 1
    
    # Compute the average distance
    return total_distance / pair_count if pair_count > 0 else 0