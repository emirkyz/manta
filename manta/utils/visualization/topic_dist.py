
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def gen_topic_dist(W,output_dir,table_name):
    """Generate a bar plot of the document distribution across topics.
    From the W matrix, first we get biggest value per row. This is the topic that the document is most associated with.
    Then we count the number of documents for each topic.
    Bar plot values should sum up to the number of documents.
    Args:
        W (numpy.ndarray): The matrix of topic distributions.
        output_dir (str): The directory to save the plot.
        table_name (str): The name of the table.
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
    
    # Check if output_dir already includes the table_name to avoid double nesting
    output_dir_path = Path(output_dir)
    if output_dir_path.name == table_name:
        table_output_dir = output_dir_path
    else:
        # Create table-specific subdirectory under output folder
        table_output_dir = output_dir_path / table_name
    table_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the plot to table-specific subdirectory
    plot_path = table_output_dir / f"{table_name}_document_dist.png"
    plt.savefig(plot_path,dpi=1000)
    print(f"Document distribution plot saved to: {plot_path}")
    return plt, topic_counts