from datasets import load_dataset
from transformers import pipeline
import numpy as np
from sklearn.cluster import DBSCAN, MeanShift
from collections import Counter
from rouge_score import rouge_scorer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer


def segment_article(article, max_segment_length, sliding_window_size):
    """
    Segments the article into overlapping windows based on the maximum segment length
    and sliding window size.

    Parameters:
    article (str): The article to be segmented.
    max_segment_length (int): The maximum length of each segment in terms of words.
    sliding_window_size (int): The size of the sliding window for overlapping segments.

    Returns:
    list: A list of segmented article chunks.
    """
    sentences = sent_tokenize(article)
    segments = []
    current_segment = []
    current_length = 0

    for i in range(len(sentences)):
        sentence_length = len(sentences[i].split())
        if current_length + sentence_length <= max_segment_length:
            current_segment.append(sentences[i])
            current_length += sentence_length
        else:
            segments.append(" ".join(current_segment))
            current_segment = sentences[i: i + sliding_window_size]
            current_length = sum(len(s.split()) for s in current_segment)

    if current_segment:
        segments.append(" ".join(current_segment))

    return segments

def sliding_generation(segments, pipe, prompt):
    """
    Generates local summaries for each segment using a language model.

    Parameters:
    segments (list): The list of segmented article chunks.
    pipe (Pipeline): The language model pipeline for text generation.
    prompt (str): The prompt to guide the summary generation.

    Returns:
    list: A list of local summaries for each segment.
    """
    local_summaries = []
    for segment in segments:
        input_text = prompt + "\n" + segment
        result = pipe(input_text, max_length=500, truncation=True, pad_token_id=pipe.tokenizer.eos_token_id)
        local_summaries.append(result[0]['generated_text'])

    return local_summaries

def remove_stopwords(sentence):
    """
    Removes stopwords from a sentence.

    Parameters:
    sentence (string): A string of sentences.

    Returns:
    string: A sentence with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sentence)
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    return ' '.join(filtered_words)

def calculate_distances(summaries):
    """
    Computes a distance matrix based on ROUGE-1 F1 scores between sentences.

    Parameters:
    summaries (list): The list of summaries.

    Returns:
    np.ndarray: A distance matrix for clustering.
    """
    sentences = [remove_stopwords(sent.strip()) for summary in summaries for sent in sent_tokenize(summary)]
    summary_sentences = [sent.strip() for summary in summaries for sent in sent_tokenize(summary)]
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    distances = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            score = scorer.score(sentences[i], sentences[j])
            rouge1_f1 = score['rouge1'].fmeasure
            distances[i, j] = distances[j, i] = 1 - rouge1_f1

    return distances, summary_sentences

def select_representative_sentence(cluster_sentences):
    """
    Select the most representative sentence from each cluster.

    Parameters:
    clusters (dict): Clusters of sentences grouped by semantic similarity.

    Returns:
    list: The most representative sentences from each cluster.
    """
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sentence_embeddings = model.encode(cluster_sentences)

    clustering = MeanShift(bandwidth=0.2, cluster_all=False).fit(sentence_embeddings)
    label_counts = Counter(clustering.labels_)
    most_common_label, _ = label_counts.most_common(1)[0]
    most_common_sentences = [cluster_sentences[idx] for idx, label in enumerate(clustering.labels_) if label == most_common_label]

    representative_sentence = max(most_common_sentences, key=len)
    
    return representative_sentence

def filter_sentences(distances, sentences, eps=0.25, min_samples=3):
    """
    Filters sentences using DBSCAN clustering to remove noise and outliers.

    Parameters:
    distances (np.ndarray): The distance matrix.
    sentences (list): The list of sentences corresponding to the distance matrix.
    eps (float): The maximum distance for two sentences to be in the same cluster.
    min_samples (int): The minimum number of sentences required to form a cluster.

    Returns:
    list: Filtered sentences representing the main points.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distances)
    filtered_sentences = []
    labels = clustering.labels_

    for label in set(labels):
        if label == -1:
            continue  # Skip noise points
        cluster_sentences = [sentences[i] for i in range(len(sentences)) if labels[i] == label]
        # Select the most frequent or longest sentence in the cluster
        filtered_sentences.append(select_representative_sentence(cluster_sentences))

    return filtered_sentences

def aggregate_summaries(filtered_sentences, pipe, prompt):
    """
    Aggregates filtered sentences into a final summary.

    Parameters:
    filtered_sentences (list): The list of filtered sentences.
    pipe (Pipeline): The language model pipeline for text generation.
    prompt (str): The prompt to guide the aggregation.

    Returns:
    str: The final aggregated summary.
    """
    final_summary = "\n".join(filtered_sentences)
    result = pipe(prompt + "\n" + final_summary, max_length=500, truncation=True)
    return result[0]['generated_text']
