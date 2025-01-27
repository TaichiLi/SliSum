import argparse
from datasets import load_dataset
from transformers import pipeline
import random
from slisum import process_sentences
from rouge_score import rouge_scorer
from bert_score import score as bertscore
import json
from tqdm import tqdm

def generate_summary_for_articles(dataset_name, model_name, window_size, step_size, distance_threshold, min_samples):
    """
    This function generates summaries for randomly selected 100 articles from the given dataset,
    evaluates the summaries using multiple metrics, and outputs the results to a JSON file.
    
    Parameters:
    - dataset_name (str): The name of the dataset to process.
    - model_name (str): The name of the model to use for summarization.
    - window_size (int): The size of the sliding window for article segmentation.
    - step_size (int): The step size for moving the sliding window.
    - distance_threshold (float): The distance threshold for DBSCAN clustering.
    - min_samples (int): The minimum number of samples required to form a cluster in DBSCAN.
    """
    
    # Load the dataset
    dataset = load_dataset(dataset_name)
    if dataset_name == "ccdv/cnn_dailymail":
        dataset_column = 'article'
        summary_column = 'highlights'
    elif dataset_name == "EdinburghNLP/xsum":
        dataset_column = 'document'
        summary_column = 'summary'
    elif dataset_name == "ccdv/arxiv-summarization":
        dataset_column = 'article'
        summary_column = 'abstract'
    elif dataset_name == "ccdv/pubmed-summarization":
        dataset_column = 'article'
        summary_column = 'abstract'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Initialize the model pipeline
    pipe = pipeline("text-generation", model=model_name, device=0)

    # Extract 100 random articles from the dataset
    articles = dataset[dataset_column]
    summaries = dataset[summary_column]
    
    # Select 100 random articles and their summaries
    random_indices = random.sample(range(len(articles)), 100)
    selected_articles = [articles[i] for i in random_indices]
    selected_summaries = [summaries[i] for i in random_indices]
    
    # Prepare evaluation metrics
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # BERTScore requires the model to compute embeddings for comparison
    results = []

    for article, ref_summary in zip(tqdm(selected_articles), selected_summaries):
        # Process each article and generate its summary
        generated_summary = process_sentences(article, pipe, window_size, step_size, distance_threshold, min_samples)
        
        # Evaluate ROUGE scores
        rouge_scores = compute_rouge_scores(generated_summary, ref_summary, rouge_scorer_instance)
        
        # Evaluate BERTScore
        bertscore_scores = compute_bertscore(generated_summary, ref_summary)
        
        # Collect the results in a dictionary
        result = {
            "article": article,
            "generated_summary": generated_summary,
            "reference_summary": ref_summary,
            "ROUGE-1": rouge_scores['rouge1'],
            "ROUGE-2": rouge_scores['rouge2'],
            "ROUGE-L": rouge_scores['rougeL'],
            "BERTScore": bertscore_scores
        }
        results.append(result)
    
    # Output results to a JSON file
    with open("evaluation_results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)


def compute_rouge_scores(generated_summary, reference_summary, scorer):
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) between generated and reference summaries.
    Uses the rouge-score library.

    Parameters:
    - generated_summary (str): The generated summary.
    - reference_summary (str): The reference summary.
    - scorer (RougeScorer): A RougeScorer instance.

    Returns:
    - dict: A dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scores = scorer.score(reference_summary, generated_summary)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def compute_bertscore(generated_summary, reference_summary):
    """
    Compute BERTScore between generated and reference summaries.
    Uses the bert-score library.

    Parameters:
    - generated_summary (str): The generated summary.
    - reference_summary (str): The reference summary.

    Returns:
    - float: The BERTScore F1 score.
    """
    P, R, F1 = bertscore([generated_summary], [reference_summary], lang='en')
    return F1[0].item()  # Returns the F1 score of the generated summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SliSum Summarization and Evaluation")

    # Define the arguments
    parser.add_argument('--model', type=str, required=True, help="Model name for summarization.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name (e.g., ccdv/cnn_dailymail, EdinburghNLP/xsum).")
    parser.add_argument('--window_size', type=int, default=150, help="Window size for sliding window.")
    parser.add_argument('--step_size', type=int, default=50, help="Step size for sliding window.")
    parser.add_argument('--distance_threshold', type=float, default=0.25, help="DBSCAN distance threshold.")
    parser.add_argument('--min_samples', type=int, default=2, help="Minimum samples for DBSCAN clustering.")
    
    args = parser.parse_args()

    # Call the function to generate summaries and evaluate them
    generate_summary_for_articles(
        dataset_name=args.dataset,
        model_name=args.model,
        window_size=args.window_size,
        step_size=args.step_size,
        distance_threshold=args.distance_threshold,
        min_samples=args.min_samples
    )
