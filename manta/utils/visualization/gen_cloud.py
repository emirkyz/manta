from wordcloud import WordCloud
from pathlib import Path
import json


def _convert_topic_name(topic_name: str) -> str:
    """Convert topic name from relevance format to display format.

    Args:
        topic_name: Topic name in format "topic_01" or "Topic 01"

    Returns:
        Standardized format "Topic 01"
    """
    if topic_name.startswith("topic_"):
        topic_num = topic_name.split("_")[1]
        return f"Topic {topic_num}"
    return topic_name


def _extract_top_words_by_relevance(word_scores: dict, top_n: int = 50) -> list:
    """Sort words by relevance score and return top N.

    Args:
        word_scores: Dictionary mapping words to relevance scores
        top_n: Number of top words to extract (default 50)

    Returns:
        List of words sorted by relevance score (highest first)
    """
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_words[:top_n]]


def _load_relevance_data(output_dir: Path, table_name: str) -> dict:
    """Load relevance_top_words.json if it exists, else return None.

    Args:
        output_dir: Directory containing the relevance JSON file
        table_name: Name of the table/dataset

    Returns:
        Dictionary of relevance data or None if not found/error
    """
    json_path = output_dir / f"{table_name}_relevance_top_words.json"
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("relevance", {})
        except Exception as e:
            print(f"Warning: Could not load relevance data: {e}")
            return None
    return None


def generate_wordclouds(topics_data, output_dir, table_name):
    """Generate wordclouds for each topic.

    Args:
        topics_data (dict): A dictionary containing topic names as keys and lists of words as values.
        output_dir (str): The directory to save the wordclouds.
        table_name (str): The name of the table.
    """
    wordclouds = {}

    # Check if output_dir already includes the table_name to avoid double nesting
    output_path = Path(output_dir)
    if output_path.name == table_name:
        table_output_dir = output_path
    else:
        # Create table-specific subdirectory under output folder
        table_output_dir = output_path / table_name

    wordclouds_dir = table_output_dir / "wordclouds"
    wordclouds_dir.mkdir(parents=True, exist_ok=True)

    # Try to load relevance data
    relevance_data = _load_relevance_data(table_output_dir, table_name)

    if relevance_data:
        # Use relevance scores for wordcloud generation
        print("Using relevance scores for wordcloud generation")
        for topic_name, word_scores in relevance_data.items():
            # Extract top words by relevance score
            words_list = _extract_top_words_by_relevance(word_scores, top_n=50)
            # Convert topic name for display
            display_name = _convert_topic_name(topic_name)

            # Generate wordcloud
            wordcloud = WordCloud(width=600, height=400, background_color='white').generate(" ".join(words_list))

            # Convert to PIL Image and save with high DPI
            image = wordcloud.to_image()
            wordclouds[display_name] = wordcloud
            image.save(str(wordclouds_dir / f"{display_name}.png"), dpi=(1000, 1000))
    else:
        # Fallback to original behavior using word_result
        print("Using word_result for wordcloud generation (relevance data not found)")
        for topic_name, words in topics_data.items():
            # Remove scores for wordcloud generation
            words_only = [word.split(":")[0] for word in words]
            wordcloud = WordCloud(width=600, height=400, background_color='white').generate(" ".join(words_only))

            # Convert to PIL Image and save with high DPI
            image = wordcloud.to_image()
            wordclouds[topic_name] = wordcloud
            image.save(str(wordclouds_dir / f"{topic_name}.png"), dpi=(1000, 1000))
