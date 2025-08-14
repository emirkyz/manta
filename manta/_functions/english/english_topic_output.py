from typing import List
import os
from pathlib import Path
from ...database.database_manager import DatabaseManager
from ...utils.image_to_base import image_to_base64
from wordcloud import WordCloud

def get_topic_names_out(data_frame_name, num_of_topics: int, sozluk: List[str], word_per_topic: int, topics_db_eng=None,
                        H=None,gen_cloud:bool = False) -> dict:
    """
    Extract topics and their top words from NMF H matrix
    
    Args:
        data_frame_name: Name of the dataset
        num_of_topics: Number of topics to extract
        sozluk: List of words in vocabulary
        word_per_topic: Number of top words to extract per topic
        topics_db_eng: SQLAlchemy engine for database connection
        H: Topic-word matrix from NMF
    """
    # Create a dictionary to store topics and their words
    topics_data = {}

    # No need to convert sozluk since it's already a list
    vocab_list = sozluk

    # For each topic
    for topic_idx in range(num_of_topics):
        topic_name = 'Konu ' + '{:02d}'.format(topic_idx + 1)

        # Get the word weights for this topic
        topic_weights = H[topic_idx]

        # Get indices of top words sorted by weight
        top_word_indices = topic_weights.argsort()[:-word_per_topic - 1:-1]
        
        # Create word:score pairs
        topic_words = []
        for idx in top_word_indices:
            word = vocab_list[idx]
            # Remove "##" prefix if it exists
            if word.startswith("##"):
                word = word[2:]
            score = float(topic_weights[idx])
            topic_words.append(f"{word}:{score:.8f}")

        topics_data[topic_name] = topic_words

    if gen_cloud:
        # Create table-specific subdirectory under Output folder
        base_dir = Path(__file__).parent.resolve()
        # Go up two levels to get to the project root, then into Output
        output_dir = base_dir / ".." / ".." / "Output"
        table_output_dir = output_dir / data_frame_name
        wordclouds_dir = table_output_dir / "wordclouds"
        wordclouds_dir.mkdir(parents=True, exist_ok=True)
        
        # generate wordcloud for each topic
        wordclouds = {}
        for topic_name, words in topics_data.items():
            # Remove scores for wordcloud generation
            words_only = [word.split(":")[0] for word in words]
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(words_only))
            a = wordcloud.to_image()

            wordclouds[topic_name] = wordcloud
            # Save wordcloud image to table-specific subdirectory
            wordcloud.to_file(str(wordclouds_dir / f"{topic_name}.png"))
            # Convert wordcloud to base64 string
            #wordcloud_base64 = image_to_base64(wordcloud)

    # Save to database if engine is provided
    if topics_db_eng:
        DatabaseManager.save_topics_to_database(topics_data, data_frame_name, topics_db_eng)
    else:
        print("Warning: No database engine provided, skipping database save")

    return topics_data


'''def get_topic_names_out(data_frame_name,model, num_of_topics: int, sozluk: List[str], word_per_topic: int, topics_db_eng=None) -> dict:
    # Create a dictionary to store topics and their words
    topics_data = {}

    # For each topic
    for i in range(num_of_topics):
        topic_name = 'Konu ' + '{:02d}'.format(i + 1)
        # Get the top words and their values
        words_ids = model.components_[i].argsort()[:-word_per_topic - 1:-1]
        words = [sozluk[key] for key in words_ids]
        values = [model.components_[i][key] for key in words_ids]

        # Store the word-value pairs for this topic
        topics_data[topic_name] = [f"{word}:{value:.8f}" for word, value in zip(words, values)]

    # Create a DataFrame where each column is a topic and each row contains one word:value pair
    max_words = max(len(words) for words in topics_data.values())
    df_data = {topic: words + [None] * (max_words - len(words))
              for topic, words in topics_data.items()}

    df = pd.DataFrame(df_data)

    # Save to database if engine is provided
    if topics_db_eng:
        table_name = f"{data_frame_name}_topics"
        df.to_sql(table_name, topics_db_eng, if_exists='replace', index=False)
    else:
        print("Warning: No database engine provided, skipping database save")

    return topics_data
'''
