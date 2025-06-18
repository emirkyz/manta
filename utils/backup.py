import json
import math
import os
from itertools import combinations
from collections import defaultdict
import numpy as np

# --- Helper Functions for Adapted Co-occurrence ---


def p_word_pair(word1,word2,documents):
    """
    Calculates the probability of a word pair in a document
    P(w1,w2) = D(w1,w2) / N
    D(w1,w2) = number of documents containing both word1 and word2
    """
    D_w1_w2 = sum(1 for doc in documents if word1 in doc and word2 in doc)
    N = len(documents)
    return D_w1_w2 / N


def p_word(word,documents):
    """
    Calculates the probability of a word in a document
    P(w) = D(w) / N
    D(w) = number of documents containing word w
    N = total number of documents
    """
    D_w = sum(1 for doc in documents if word in doc)
    N = len(documents)
    return D_w / N

def pmi(word1,word2,documents,epsilon=1e-9):
    """
    Calculates the probability of a word pair in a document
    P(w1,w2) = D(w1,w2) / N
    D(w1,w2) = number of documents containing both word1 and word2
    PMI(w1,w2) = log(P(w1,w2) / (P(w1) * P(w2)))
    """

    p1 = p_word(word1,documents)
    p2 = p_word(word2,documents)
    if p1 == 0 or p2 == 0:
        return "zero_division_error"
    return math.log((p_word_pair(word1,word2,documents) + epsilon) / (p1 * p2))


def c_uci(documents,topics_json,epsilon=1e-9):
    """
    Calculates the UCI coherence score for a topic
    UCI(w1,w2) = 2 / (N * (N-1)) * sum_i sum_j PMI(w_i,w_j)
    """
    total_topic_count = len(topics_json)
    if total_topic_count == 0:
        print("Error: No topics found in the data.")
        return None

    topic_coherences = {}
    total_coherence_sum = 0
    valid_topics_count = 0
    for i in range(len(topics_json)):
        topic = topics_json[f"topic_{i}"]
        N = len(topic)
        word_combinations = combinations(topic.keys(),2)
        current_topic_pmi_sum = 0
        for word1, word2 in word_combinations:
            pmi_val = pmi(word1, word2, documents, epsilon)
            if pmi_val == "zero_division_error":
                continue
            current_topic_pmi_sum += pmi_val
        topic_coherences[f"topic_{i}_coherence"] = current_topic_pmi_sum / N
        total_coherence_sum += topic_coherences[f"topic_{i}_coherence"]
        valid_topics_count += 1

    average_coherence = total_coherence_sum / valid_topics_count if valid_topics_count > 0 else 0.0
    return {"topic_coherences": topic_coherences, "average_coherence": average_coherence}




# --- Example Usage ---
if __name__ == '__main__':
    # Example JSON data as a string
    # Simulating a scenario where words co-occur across topics
    example_topics_json = """
    {
      "topic_0": {
        "apple": 0.5,
        "banana": 0.4,
        "fruit": 0.6,
        "healthy": 0.3
      },
      "topic_1": {
        "banana": 0.7,
        "fruit": 0.5,
        "healthy": 0.8,
        "smoothie": 0.6
      },
      "topic_2": {
        "car": 0.9,
        "road": 0.8,
        "travel": 0.7
      },
      "topic_3": {
        "apple": 0.6,
        "fruit": 0.7,
        "pie": 0.5
      }
    }
    """

    print("--- Calculating C_uci ---")
    total_topics = len(json.loads(example_topics_json))
    imp_results = {}
    for topic in json.loads(example_topics_json):
        print(f"Topic {topic} has {len(json.loads(example_topics_json)[topic])} words.")
        imp_results[topic] = c_uci(json.loads(example_topics_json)[topic],json.loads(example_topics_json))
    print(imp_results)
    
    if imp_results:
        print("Topic C_uci Coherences:", json.dumps(imp_results["topic_coherences"], indent=2))
        print("Average C_uci Coherence:", imp_results["average_coherence"])
    print("\n")

    print("--- Calculating C_npmi ---")
    c_npmi_results = calculate_c_npmi(json.loads(example_topics_json))
    if c_npmi_results:
        print("Topic C_npmi Coherences:", json.dumps(c_npmi_results["topic_coherences"], indent=2))
        print("Average C_npmi Coherence:", c_npmi_results["average_coherence"])
    print("\n")

    print("--- Calculating C_umass ---")
    c_umass_results = calculate_c_umass(json.loads(example_topics_json))
    if c_umass_results:
        print("Topic C_umass Coherences:", json.dumps(c_umass_results["topic_coherences"], indent=2))
        print("Average C_umass Coherence:", c_umass_results["average_coherence"])
    print("\n")

    print("--- Calculating C_v ---")
    c_v_results = calculate_c_v(json.loads(example_topics_json))
    if c_v_results:
        print("Topic C_v Coherences:", json.dumps(c_v_results["topic_coherences"], indent=2))
        print("Average C_v Coherence:", c_v_results["average_coherence"])

    # Example with a single topic (some scores might be 0 or behave differently)
    single_topic_json = """
    {
      "topic_0": {
        "computer": 0.8,
        "science": 0.7,
        "code": 0.9
      }
    }
    """
    print("\n--- Single Topic Example (C_npmi) ---")
    # For a single topic, D(w) and D(w1,w2) will be 1 if present, N=1.
    # P(w) = 1, P(w1,w2) = 1.
    # PMI = log(1/(1*1)) = 0. NPMI = 0 / -log(1) -> undefined or handled as 0 or 1.
    # The _calculate_corpus_statistics will reflect this.
    # D(w1)=1, D(w2)=1, D(w1,w2)=1. N=1.
    # prob_w1 = 1, prob_w2 = 1, prob_w1_w2 = 1.
    # NPMI will be tricky, as -log(prob_w1_w2) would be -log(1) = 0.
    # This is an edge case of the adapted statistics.
    # Gensim handles NPMI when P(u,v)=1 and P(u)=1, P(v)=1 to be 1.0 (for C_V).
    # My NPMI calculation for this case (prob_w1_w2 == 1.0) gives npmi = 1.0.
    c_npmi_single = calculate_c_npmi(json.loads(single_topic_json))
    if c_npmi_single:
        print("Topic C_npmi Coherences:", json.dumps(c_npmi_single["topic_coherences"], indent=2))
        print("Average C_npmi Coherence:", c_npmi_single["average_coherence"])

    # Example with words that don't co-occur
    no_cooccurrence_json = """
    {
      "topic_0": {"cat": 1.0, "dog": 1.0},
      "topic_1": {"fish": 1.0, "bird": 1.0}
    }
    """
    # Here, ("cat", "dog") co-occur in topic_0. D("cat","dog")=1.
    # D("cat")=1, D("dog")=1, D("fish")=1, D("bird")=1. N=2.
    # P("cat","dog") = 1/2. P("cat")=1/2, P("dog")=1/2.
    # PMI("cat","dog") = log( (1/2) / (1/2 * 1/2) ) = log(2).
    # NPMI("cat","dog") = log(2) / -log(1/2) = log(2) / log(2) = 1.
    print("\n--- No Global Co-occurrence Example (C_npmi) ---")
    c_npmi_no_co = calculate_c_npmi(json.loads(no_cooccurrence_json))
    if c_npmi_no_co:
        print("Topic C_npmi Coherences:", json.dumps(c_npmi_no_co["topic_coherences"], indent=2))
        print("Average C_npmi Coherence:", c_npmi_no_co["average_coherence"])

def calculate_coherence_scores(topic_word_scores, output_dir=None, table_name=None):
    """
    Calculate all coherence scores (UCI, NPMI, UMass, C_v) and optionally save them.

    Args:
        topic_word_scores (dict): Dictionary of topics and their word scores
        output_dir (str, optional): Directory to save the scores. If None, scores won't be saved
        table_name (str, optional): Name of the table/analysis for file naming

    Returns:
        dict: Dictionary containing all coherence scores
    """
    print("Calculating coherence scores...")
    coherence_scores = {
        "c_uci": calculate_c_uci(topic_word_scores),
        "c_npmi": calculate_c_npmi(topic_word_scores),
        "c_umass": calculate_c_umass(topic_word_scores),
        "c_v": calculate_c_v(topic_word_scores)
    }

    # Print coherence scores
    print("\nCoherence Scores:")
    for metric, scores in coherence_scores.items():
        if scores:  # Check if scores were calculated successfully
            print(f"{metric.upper()}: {scores['average_coherence']:.4f}")

    # Save coherence scores if output directory and table name are provided
    if output_dir and table_name:
        coherence_file = os.path.join(output_dir, table_name, f"{table_name}_coherence_scores.json")
        os.makedirs(os.path.dirname(coherence_file), exist_ok=True)
        with open(coherence_file, "w") as f:
            json.dump(coherence_scores, f, indent=4)
        print("Coherence scores saved to:", coherence_file)

    return coherence_scores
'''
import json
import math
import os
from itertools import combinations
from collections import defaultdict
import numpy as np

# --- Helper Functions for Adapted Co-occurrence ---



def p_word_pair(word1,word2,documents):
    """
    Calculates the probability of a word pair in a document
    P(w1,w2) = D(w1,w2) / N
    D(w1,w2) = number of documents containing both word1 and word2
    """
    D_w1_w2 = sum(1 for doc in documents if word1 in doc and word2 in doc)
    N = len(documents)
    return D_w1_w2 / N


def p_word(word,documents):
    """
    Calculates the probability of a word in a document
    P(w) = D(w) / N
    D(w) = number of documents containing word w
    N = total number of documents
    """
    D_w = sum(1 for doc in documents if word in doc)
    N = len(documents)
    return D_w / N

def pmi(word1,word2,documents,epsilon=1e-9):
    """
    Calculates the probability of a word pair in a document
    P(w1,w2) = D(w1,w2) / N
    D(w1,w2) = number of documents containing both word1 and word2
    PMI(w1,w2) = log(P(w1,w2) / (P(w1) * P(w2)))
    """

    p1 = p_word(word1,documents)
    p2 = p_word(word2,documents)
    if p1 == 0 or p2 == 0:
        return "zero_division_error"
    return math.log((p_word_pair(word1,word2,documents) + epsilon) / (p1 * p2))


def c_uci(topics_json, documents=None, epsilon=1e-9):
    """
    Calculates the UCI coherence score for topics
    UCI(w1,w2) = 2 / (N * (N-1)) * sum_i sum_j PMI(w_i,w_j)
    
    Args:
        topics_json (dict): Dictionary containing topics and their word scores
        documents (list, optional): List of documents for co-occurrence calculation. 
                                  If None, will use topics themselves as documents.
        epsilon (float): Small value to prevent log(0)
        
    Returns:
        dict: Dictionary containing topic coherences and average coherence
    """
    # If no documents provided, create pseudo-documents from topics
    if documents is None:
        documents = []
        for topic_id, word_scores in topics_json.items():
            # Create a document containing all words from this topic
            doc = list(word_scores.keys())
            documents.append(doc)
    
    total_topic_count = len(topics_json)
    if total_topic_count == 0:
        print("Error: No topics found in the data.")
        return None

    topic_coherences = {}
    total_coherence_sum = 0
    valid_topics_count = 0
    
    for topic_id, word_scores in topics_json.items():
        # Sort words by their scores in descending order and take top words
        sorted_words = sorted(word_scores.items(), key=lambda x: float(x[1]), reverse=True)
        top_words = [word for word, _ in sorted_words]
        
        N = len(top_words)
        if N < 2:  # Need at least 2 words to calculate coherence
            continue
            
        word_combinations = combinations(top_words, 2)
        pmi_values = []
        
        for word1, word2 in word_combinations:
            pmi_val = pmi(word1, word2, documents, epsilon)
            if pmi_val != "zero_division_error":
                pmi_values.append(pmi_val)
        
        if pmi_values:  # Only calculate if we have valid PMI values
            # Calculate UCI coherence for this topic
            topic_coherence = 2 * sum(pmi_values) / (N * (N - 1))
            topic_coherences[f"{topic_id}_coherence"] = topic_coherence
            total_coherence_sum += topic_coherence
            valid_topics_count += 1

    average_coherence = total_coherence_sum / valid_topics_count if valid_topics_count > 0 else 0.0
    return {
        "topic_coherences": topic_coherences, 
        "average_coherence": average_coherence
    }




# --- Example Usage ---
if __name__ == '__main__':
    # Load the example wordcloud scores
    with open('../Output/FINDINGS_pnmf/FINDINGS_pnmf_wordcloud_scores.json', 'r') as f:
        wordcloud_scores = json.load(f)
    
    # Calculate coherence scores
    coherence_results = c_uci(wordcloud_scores)
    print("\n--- UCI Coherence Scores ---")
    print(f"Average Coherence: {coherence_results['average_coherence']:.4f}")
    print("\nPer-topic Coherence Scores:")
    for topic, score in coherence_results['topic_coherences'].items():
        print(f"{topic}: {score:.4f}")

def calculate_coherence_scores(topic_word_scores, output_dir=None, table_name=None):
    """
    Calculate coherence scores (UCI) and optionally save them.

    Args:
        topic_word_scores (dict): Dictionary of topics and their word scores
        output_dir (str, optional): Directory to save the scores. If None, scores won't be saved
        table_name (str, optional): Name of the table/analysis for file naming

    Returns:
        dict: Dictionary containing coherence scores
    """
    print("Calculating coherence scores...")
    
    # Calculate UCI coherence
    coherence_scores = c_uci(topic_word_scores)
    
    # Print coherence scores
    print("\nCoherence Scores:")
    print(f"UCI Average Coherence: {coherence_scores['average_coherence']:.4f}")
    print("\nPer-topic Coherence Scores:")
    for topic, score in coherence_scores['topic_coherences'].items():
        print(f"{topic}: {score:.4f}")

    # Save coherence scores if output directory and table name are provided
    if output_dir and table_name:
        coherence_file = os.path.join(output_dir, f"{table_name}_coherence_scores.json")
        os.makedirs(os.path.dirname(coherence_file), exist_ok=True)
        with open(coherence_file, "w") as f:
            json.dump(coherence_scores, f, indent=4)
        print(f"\nCoherence scores saved to: {coherence_file}")

    return coherence_scores



'''