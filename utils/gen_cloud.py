from wordcloud import WordCloud
import os

def generate_wordclouds(topics_data,output_dir,table_name):
    """
    Generate and save word cloud images for each topic.
    
    Args:
        topics_data (dict): Dictionary mapping topic names to lists of words with scores
        output_dir (str): Base output directory path
        table_name (str): Analysis identifier for subdirectory naming
    
    Returns:
        None: Saves word cloud images to disk
    """
    wordclouds = {}
    
    # Create table-specific subdirectory under output folder
    table_output_dir = os.path.join(output_dir, table_name)
    os.makedirs(table_output_dir+"/wordclouds", exist_ok=True)
    for topic_name, words in topics_data.items():
        # Remove scores for wordcloud generation
        words_only = [word.split(":")[0] for word in words]
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(words_only))
        a = wordcloud.to_image()

        wordclouds[topic_name] = wordcloud
        # Save wordcloud image to table-specific subdirectory
        wordcloud.to_file(f"{table_output_dir}/wordclouds/{topic_name}.png")
  