import argparse
import random
from datasets import load_dataset
import json


def get_articles_within_word_count(
    dataset_name, target_length, num_articles, tolerance=0.05
):
    """
    Retrieves articles and their reference summaries from the specified dataset with a word count
    within the target length plus or minus a specified tolerance.

    Parameters:
    - dataset_name (str): Name of the dataset (e.g., 'ccdv/cnn_dailymail').
    - target_length (int): The target word length for the articles.
    - num_articles (int): Number of articles to retrieve.
    - tolerance (float): Tolerance range for word count (default is 5%).

    Returns:
    - list: A list of dictionaries with articles and their reference summaries.
    """
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Identify the column names for the article and reference summary
    if dataset_name == "ccdv/cnn_dailymail":
        article_column = "article"
        summary_column = "highlights"
        id_column_available = True
    elif dataset_name == "EdinburghNLP/xsum":
        article_column = "document"
        summary_column = "summary"
        id_column_available = True
    elif dataset_name == "ccdv/arxiv-summarization":
        article_column = "article"
        summary_column = "abstract"
        id_column_available = False
    elif dataset_name == "ccdv/pubmed-summarization":
        article_column = "article"
        summary_column = "abstract"
        id_column_available = False
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Get all articles and summaries
    articles = dataset[article_column]
    summaries = dataset[summary_column]

    # If no ID column is available, skip IDs
    if id_column_available:
        ids = dataset["id"]
    else:
        ids = [None] * len(articles)  # Use None when IDs are not present

    # Calculate the acceptable word count range
    min_word_count = target_length * (1 - tolerance)
    max_word_count = target_length * (1 + tolerance)

    # Filter articles based on word count within the tolerance range
    filtered_articles = []
    for i in range(len(articles)):
        word_count = len(articles[i].split())
        if min_word_count <= word_count <= max_word_count:
            if ids[i] is not None:
                filtered_articles.append(
                    {
                        "id": ids[i],
                        article_column: articles[i],
                        summary_column: summaries[i],
                    }
                )
            else:
                filtered_articles.append(
                    {article_column: articles[i], summary_column: summaries[i]}
                )

    # Randomly sample the required number of articles
    sampled_articles = random.sample(filtered_articles, num_articles)

    return sampled_articles


def save_articles_to_json(articles, output_file):
    """
    Save the selected articles and their reference summaries to a JSON file.

    Parameters:
    - articles (list): List of articles with optional IDs, content, and reference summaries.
    - output_file (str): The path of the output JSON file.
    """
    with open(output_file, "w") as json_file:
        json.dump(articles, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get articles with specific word count range and save to JSON."
    )

    # Define the arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., ccdv/cnn_dailymail).",
    )
    parser.add_argument(
        "--target_length",
        type=int,
        required=True,
        help="Target word length for articles.",
    )
    parser.add_argument(
        "--num_articles",
        type=int,
        required=True,
        help="Number of articles to retrieve.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Tolerance range for word count (default is 5%).",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output JSON file name."
    )

    args = parser.parse_args()

    # Get the articles within the specified word count range
    selected_articles = get_articles_within_word_count(
        dataset_name=args.dataset,
        target_length=args.target_length,
        num_articles=args.num_articles,
        tolerance=args.tolerance,
    )

    # Save the selected articles to a JSON file
    save_articles_to_json(selected_articles, args.output_file)
