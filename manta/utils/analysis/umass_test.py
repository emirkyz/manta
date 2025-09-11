
from collections import defaultdict
import math


class UMassCoherence:
    def __init__(self, documents, epsilon=1):
        """
        Initialize U_mass coherence calculator

        Args:
            documents: List of documents, where each document is a list of words
            epsilon: Smoothing parameter to avoid log(0), typically 1
                    Gensim uses 1 as default, which is standard
        """
        self.documents = documents
        self.epsilon = epsilon
        self.word_doc_freq = defaultdict(set)
        self.cooccur_freq = defaultdict(lambda: defaultdict(int))

        # Build word-document frequency and co-occurrence matrices
        self._build_frequencies()

    def _build_frequencies(self):
        """Build word-document frequency and co-occurrence frequency dictionaries"""
        for doc_id, doc in enumerate(self.documents):
            # Get unique words in document
            unique_words = set(doc)

            # Track which documents contain each word
            for word in unique_words:
                self.word_doc_freq[word].add(doc_id)

            # Track co-occurrences
            unique_words_list = list(unique_words)
            for i in range(len(unique_words_list)):
                for j in range(i + 1, len(unique_words_list)):
                    word1, word2 = unique_words_list[i], unique_words_list[j]
                    # Store co-occurrences symmetrically
                    self.cooccur_freq[word1][word2] += 1
                    self.cooccur_freq[word2][word1] += 1

    def calculate_umass_coherence(self, topic_words, top_n=10):
        """
        Calculate U_mass coherence for a topic

        Args:
            topic_words: List of (word, score) tuples for the topic
            top_n: Number of top words to consider

        Returns:
            U_mass coherence score (average of all pairwise scores)

        Implementation notes:
        - Words should be ordered by decreasing probability/score in the topic
        - Formula: sum over all i<j of log((D(wi,wj) + ε) / D(wj))
          where wi is less frequent than wj in the topic
        - Returns the arithmetic mean (average) of pairwise scores
        - Higher scores (closer to 0) indicate better coherence
        - Typical range: -14 to 14, but can exceed these bounds
        """
        # Get top N words
        if isinstance(topic_words, dict):
            # Sort by score and get top N
            sorted_words = sorted(topic_words.items(), key=lambda x: x[1], reverse=True)
            top_words = [word for word, _ in sorted_words[:top_n]]
        else:
            # Assume it's already a list of words
            top_words = topic_words[:top_n]

        coherence_score = 0.0
        pair_count = 0

        # Calculate pairwise coherence
        # Loop structure ensures i > j, meaning word_i has lower probability than word_j
        # This matches U_mass requirement: we calculate P(less_frequent | more_frequent)
        for i in range(1, len(top_words)):
            for j in range(i):
                word_i = top_words[i]  # Less frequent word (lower probability in topic)
                word_j = top_words[j]  # More frequent word (higher probability in topic)

                # Get document frequencies
                D_wi = len(self.word_doc_freq.get(word_i, set()))
                D_wj = len(self.word_doc_freq.get(word_j, set()))

                # Get co-occurrence frequency
                D_wi_wj = self.cooccur_freq.get(word_i, {}).get(word_j, 0)

                # Calculate U_mass score for this pair
                if D_wj > 0:  # Only need to check denominator
                    # U_mass formula: log((D(wi, wj) + epsilon) / D(wj))
                    # Note: wi is less frequent, wj is more frequent due to loop structure
                    score = math.log((D_wi_wj + self.epsilon) / D_wj)
                    coherence_score += score
                    pair_count += 1

        # Return average of all pairwise scores (arithmetic mean)
        if pair_count > 0:
            return coherence_score / pair_count
        else:
            return 0.0

    def calculate_all_topics_coherence(self, topics_dict, top_n=10):
        """
        Calculate U_mass coherence for all topics

        Args:
            topics_dict: Dictionary of topics with word scores
            top_n: Number of top words to consider per topic

        Returns:
            Dictionary of coherence scores for each topic
        """
        coherence_scores = {}

        for topic_name, topic_words in topics_dict.items():
            coherence_scores[topic_name] = self.calculate_umass_coherence(topic_words, top_n)

        # Calculate average coherence across all topics
        if coherence_scores:
            average_coherence = sum(coherence_scores.values()) / len(coherence_scores)
        else:
            average_coherence = 0.0

        return coherence_scores, average_coherence


# Example usage with the provided data
if __name__ == "__main__":
    # Sample documents
    sample_documents = [
        ["machine", "learning", "algorithm", "data", "science", "model", "prediction"],
        ["neural", "network", "deep", "learning", "artificial", "intelligence"],
        ["natural", "language", "processing", "text", "nlp", "analysis"],
        ["machine", "learning", "model", "training", "data", "algorithm"],
        ["deep", "neural", "network", "artificial", "intelligence", "learning"],
        ["text", "processing", "natural", "language", "nlp", "analysis"],
        ["data", "science", "machine", "learning", "model", "algorithm"],
        ["artificial", "intelligence", "neural", "network", "deep"],
        ["language", "processing", "text", "natural", "nlp"],
        ["learning", "machine", "algorithm", "data", "model"]
    ]

    # Your Turkish topics from the JSON
    topics_dict = {
        "Konu 00": {
            "şarkı": 3.82452107,
            "bir": 3.27153271,
            "reklam": 2.34486783,
            "uygulama / uygulamayı": 2.03144656,
            "müzik": 1.97318196,
            "bi": 1.962547,
            "1": 1.68050699,
            "30": 1.65562268,
            "kendi": 1.64953604,
            "yok": 1.64099496,
            "şarkılar": 1.63648208,
            "bile": 1.46386883,
            "aynı": 1.45637245,
            "2": 1.42662329,
            "sorun": 1.42400974
        },
        "Konu 01": {
            "sonra": 6.2857178,
            "son": 5.50632777,
            "açılmıyor": 1.3237279,
            "atıyor": 0.86620733,
            "iphone": 0.6657557,
            "başladı": 0.66319478,
            "6": 0.6291712,
            "kapanıyor": 0.5854877,
            "uygulamadan": 0.54826726,
            "güncellemeden / guncellemeden": 0.52424671,
            "çalışmıyor": 0.4904359,
            "müzik": 0.43832109,
            "oldu": 0.42959698,
            "acil": 0.41326912,
            "ios": 0.40779857
        }
        # Add more topics as needed
    }

    # Initialize the coherence calculator
    coherence_calc = UMassCoherence(sample_documents)

    # Example: Calculate coherence for a single topic using sample English words
    sample_topic = ["machine", "learning", "algorithm", "data", "model"]
    single_coherence = coherence_calc.calculate_umass_coherence(sample_topic)
    print(f"Sample topic coherence: {single_coherence:.4f}")

    # Detailed example showing the calculation
    print("\nDetailed calculation for topic ['machine', 'learning', 'algorithm']:")
    topic_words = ["machine", "learning", "algorithm"]
    score_sum = 0
    pair_count = 0
    for i in range(1, len(topic_words)):
        for j in range(i):
            wi = topic_words[i]
            wj = topic_words[j]
            D_wi = len(coherence_calc.word_doc_freq.get(wi, set()))
            D_wj = len(coherence_calc.word_doc_freq.get(wj, set()))
            D_both = coherence_calc.cooccur_freq.get(wi, {}).get(wj, 0)
            if D_wj > 0:
                pair_score = math.log((D_both + 1) / D_wj)
                score_sum += pair_score
                pair_count += 1
                print(f"  P({wi}|{wj}) = log(({D_both}+1)/{D_wj}) = {pair_score:.4f}")
    avg_score = score_sum / pair_count if pair_count > 0 else 0
    print(f"  Sum of scores: {score_sum:.4f}")
    print(f"  Number of pairs: {pair_count}")
    print(f"  Average U_mass score: {avg_score:.4f}")

    # For your Turkish topics, you would need Turkish documents
    # Here's how you would use it:
    print("\nTo use with your Turkish topics:")
    print("1. Load your Turkish documents (each as a list of words)")
    print("2. Initialize: coherence_calc = UMassCoherence(turkish_documents)")
    print("3. Calculate: scores, avg_score = coherence_calc.calculate_all_topics_coherence(topics_dict, top_n=10)")
    print("4. Access individual scores: scores['Konu 00']")
    print("5. Access average score: avg_score")

    # Demonstrate the calculation process
    print("\nDemonstration of U_mass calculation:")
    print("For words 'machine' and 'learning':")
    D_machine = len(coherence_calc.word_doc_freq.get("machine", set()))
    D_learning = len(coherence_calc.word_doc_freq.get("learning", set()))
    D_both = coherence_calc.cooccur_freq.get("machine", {}).get("learning", 0)
    print(f"  D(machine) = {D_machine}")
    print(f"  D(learning) = {D_learning}")
    print(f"  D(machine, learning) = {D_both}")

    # In our implementation, 'learning' appears before 'machine' in frequency order
    # So we calculate P(machine|learning) = D(machine,learning) / D(learning)
    if D_both > 0 and D_learning > 0:
        umass_score = math.log((D_both + 1) / D_learning)
        print(f"  U_mass = log(({D_both}+1)/{D_learning}) = {umass_score:.4f}")

    # Complete working example for calculating coherence
    print("\n" + "=" * 60)
    print("COMPARISON WITH GENSIM:")
    print("=" * 60)
    print("This implementation follows Gensim's approach:")
    print("- Uses arithmetic_mean aggregation (averages pairwise scores)")
    print("- Epsilon smoothing parameter = 1")
    print("- Document-level co-occurrence (not sliding window)")
    print("- Returns average of log conditional probabilities")
    print("\nFor exact compatibility with Gensim, ensure:")
    print("- Same text preprocessing (tokenization, lowercasing)")
    print("- Same vocabulary (no out-of-vocabulary words)")
    print("- Same document structure")

    print("\n" + "=" * 60)
    print("COMPLETE WORKING EXAMPLE:")
    print("=" * 60)

    # Example: If you have your documents loaded
    # turkish_documents = [
    #     ["şarkı", "müzik", "dinle", "uygulama"],
    #     ["reklam", "çok", "var", "rahatsız"],
    #     # ... more documents
    # ]

    # Example usage pattern:
    print("\n# Initialize with your corpus")
    print("coherence_calc = UMassCoherence(turkish_documents)")
    print("\n# Calculate coherence for all topics")
    print("coherence_scores, average_score = coherence_calc.calculate_all_topics_coherence(topics_dict, top_n=10)")
    print("\n# Display results")
    print("print(f'Average coherence score: {average_score:.4f}')")
    print("for topic_name, score in coherence_scores.items():")
    print("    print(f'{topic_name}: {score:.4f}')")
    print("\nTypical U_mass score ranges:")
    print("- Good coherence: -2 to -6")
    print("- Average coherence: -6 to -10")
    print("- Poor coherence: -10 to -14 or lower")
    print("\nIf scores are unusually high (close to 0 or positive):")
    print("- Check corpus size (very small corpus can cause high scores)")
    print("- Verify word preprocessing matches between corpus and topics")
    print("- Ensure documents contain diverse vocabulary")