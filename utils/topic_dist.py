import os
import numpy as np
import matplotlib.pyplot as plt

def gen_topic_dist(W,output_dir,table_name):
    """
    Generate and save a bar plot showing document distribution across topics.
    
    Args:
        W (numpy.ndarray): Document-topic matrix from NMF decomposition
        output_dir (str): Base output directory path
        table_name (str): Analysis identifier for file naming
    
    Returns:
        None: Saves distribution plot to disk
    """
    print("Calculating document distribution across topics...")
    dominant_topics = np.argmax(W, axis=1)
    # Count number of documents per topic
    topic_counts = np.bincount(dominant_topics)
    
    # Print the counts
    print("\nNumber of documents per topic:")
    for topic_idx, count in enumerate(topic_counts):
        print(f"Topic {topic_idx + 1}: {count} documents")

    start_index = 1
    end_index = len(topic_counts) + 1
    # Create and save bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(start_index, end_index), topic_counts)
    plt.xlabel('Topic Number')
    plt.ylabel('Number of Documents')
    plt.title('Number of Documents per Topic')
    plt.xticks(range(start_index, end_index))
    
    # Add count labels on top of each bar
    for i, count in enumerate(topic_counts):
        plt.text(i+start_index, count, str(count), ha='center', va='bottom')
    
    # Create table-specific subdirectory under output folder
    table_output_dir = os.path.join(output_dir, table_name)
    os.makedirs(table_output_dir, exist_ok=True)
    
    # Save the plot to table-specific subdirectory
    plot_path = os.path.join(table_output_dir, f"{table_name}_document_dist.png")
    plt.savefig(plot_path)
    plt.close()
    