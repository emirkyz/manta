from wordcloud import WordCloud
from pathlib import Path


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

    for topic_name, words in topics_data.items():
        # Remove scores for wordcloud generation
        words_only = [word.split(":")[0] for word in words]
        wordcloud = WordCloud(width=600, height=400, background_color='white').generate(" ".join(words_only))

        # Convert to PIL Image and save with high DPI
        image = wordcloud.to_image()

        wordclouds[topic_name] = wordcloud
        # Save wordcloud image to table-specific subdirectory with 400 DPI
        image.save(str(wordclouds_dir / f"{topic_name}.png"),dpi=(1000,1000))
        # TODO: Re-add DPI setting if needed, WordCloud does not support DPI directly , dpi=(1000, 1000)
