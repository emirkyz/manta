from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

import manta
from manta._functions.english.english_preprocessor import clean_english_text
from manta._functions.english.english_text_encoder import counterize_english
from manta._functions.tfidf import tf_l


def load_model_components(model_file="topic_model_components.npz"):
    """Load saved model components from npz file."""
    try:
        data = np.load(model_file, allow_pickle=True)
        W = data['W']
        H = data['H']
        vocab = data['vocab']

        # Handle different vocab formats - convert to list if needed
        if isinstance(vocab, np.ndarray):
            if vocab.ndim == 0:  # scalar array (single item)
                vocab = vocab.item()
            else:  # array of items
                vocab = vocab.tolist()

        print(f"Model components loaded from {model_file}")
        print(f"Vocabulary type: {type(vocab)}, length: {len(vocab) if hasattr(vocab, '__len__') else 'N/A'}")
        return W, H, vocab
    except FileNotFoundError:
        print(f"Model file {model_file} not found. Run training first.")
        return None, None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None


def predict_topics(text, model_file="topic_model_components.npz", top_n=3, normalization='l1'):
    """
    Predict topics for new text using saved model components.

    Args:
        text (str): Input text to analyze
        model_file (str): Path to saved model components
        top_n (int): Number of top topics to return
        normalization (str): Normalization method - 'l1', 'l2', 'minmax', 'softmax', or 'none'

    Returns:
        dict: Dictionary with raw scores and normalized scores using specified method
    """
    # Load model components
    W, H, vocab = load_model_components(model_file)
    if H is None or vocab is None:
        return None

    # Preprocess the input text
    sentence = clean_english_text([text], lemmatize=True)
    data = counterize_english(vocab, sentence, lemmatize=True)

    if not data or not data[0]:  # Check if preprocessing resulted in empty data
        print("No valid tokens found in input text")
        return None

    # Create document-term matrix
    document_count = len(data)
    vocabulary_count = len(vocab)
    matris = lil_matrix((document_count, vocabulary_count), dtype=int)

    for i, document in enumerate(data):
        histogram = Counter(document)
        temporary = [(k, v) for k, v in histogram.items()]
        columns = [a[0] for a in temporary]
        values = [b[1] for b in temporary]
        matris[i, columns] = values

    # Calculate TF-IDF
    input_matrix = matris.tocsc(copy=True)
    input_matrix.data = np.ones_like(input_matrix.data)

    idf = 1
    tf = tf_l(input_matrix)
    tf_idf = tf.multiply(idf)

    # Normalize
    tf_idf = tf_idf.tocsr()
    from sklearn.preprocessing import normalize
    tf_idf = normalize(tf_idf, norm='l2', axis=1)

    # Predict topic distribution
    raw_scores = H @ tf_idf.T

    # Apply L1 normalization (better alternative to softmax for topic modeling)
    raw_scores_flat = raw_scores.flatten()  # Convert to 1D array for easier processing
    l1_scores = np.abs(raw_scores_flat) / np.sum(np.abs(raw_scores_flat))

    # Get top topics for raw scores
    raw_labeled_results = [(f"Topic {i+1}", raw_scores_flat[i]) for i in range(len(raw_scores_flat))]
    raw_sorted_results = sorted(raw_labeled_results, key=lambda x: x[1], reverse=True)

    # Get top topics for L1 normalized scores
    l1_labeled_results = [(f"Topic {i+1}", l1_scores[i]) for i in range(len(l1_scores))]
    l1_sorted_results = sorted(l1_labeled_results, key=lambda x: x[1], reverse=True)

    return {
        'raw_scores': raw_sorted_results[:top_n],
        'l1_scores': l1_sorted_results[:top_n]
    }


def predict_topics_with_components(text, W, H, vocab, top_n=3, normalization='l1'):
    """
    Predict topics for new text using pre-loaded model components.

    Args:
        text (str): Input text to analyze
        W: Topic-word matrix from NMF
        H: Document-topic matrix from NMF
        vocab: Vocabulary list
        top_n (int): Number of top topics to return
        normalization (str): Normalization method - 'l1', 'l2', 'minmax', 'softmax', or 'none'

    Returns:
        dict: Dictionary with raw scores and normalized scores using specified method
    """
    # Preprocess the input text
    sentence = clean_english_text([text], lemmatize=True)
    data = counterize_english(vocab, sentence, lemmatize=True)

    if not data or not data[0]:  # Check if preprocessing resulted in empty data
        print("No valid tokens found in input text")
        return None

    # Create document-term matrix
    document_count = len(data)
    vocabulary_count = len(vocab)
    matris = lil_matrix((document_count, vocabulary_count), dtype=int)

    for i, document in enumerate(data):
        histogram = Counter(document)
        temporary = [(k, v) for k, v in histogram.items()]
        columns = [a[0] for a in temporary]
        values = [b[1] for b in temporary]
        matris[i, columns] = values

    # Calculate TF-IDF
    input_matrix = matris.tocsc(copy=True)
    input_matrix.data = np.ones_like(input_matrix.data)

    idf = 1
    tf = tf_l(input_matrix)
    tf_idf = tf.multiply(idf)

    # Normalize
    tf_idf = tf_idf.tocsr()
    from sklearn.preprocessing import normalize
    tf_idf = normalize(tf_idf, norm='l2', axis=1)

    # Predict topic distribution
    raw_scores = H @ tf_idf.T

    # Apply L1 normalization (better alternative to softmax for topic modeling)
    raw_scores_flat = raw_scores.flatten()  # Convert to 1D array for easier processing
    l1_scores = np.abs(raw_scores_flat) / np.sum(np.abs(raw_scores_flat))

    # Get top topics for raw scores
    raw_labeled_results = [(f"Topic {i+1}", raw_scores_flat[i]) for i in range(len(raw_scores_flat))]
    raw_sorted_results = sorted(raw_labeled_results, key=lambda x: x[1], reverse=True)

    # Get top topics for L1 normalized scores
    l1_labeled_results = [(f"Topic {i+1}", l1_scores[i]) for i in range(len(l1_scores))]
    l1_sorted_results = sorted(l1_labeled_results, key=lambda x: x[1], reverse=True)

    return {
        'raw_scores': raw_sorted_results[:top_n],
        'l1_scores': l1_sorted_results[:top_n]
    }


def predict_topics_for_processed_data(processed_data, W, H, vocab):
    """
    Predict topics for pre-processed data and return prediction matrix and topic counts.

    Args:
        processed_data: Pre-processed text array from manta process
        W: Topic-document matrix from NMF training
        H: Topic-word matrix from NMF training
        vocab: Vocabulary list

    Returns:
        dict: Dictionary containing prediction matrix and topic counts from processed_data predictions
    """
    num_topics = H.shape[0]
    topic_labels = [f"Topic {i+1}" for i in range(num_topics)]
    num_documents = len(processed_data)

    # Initialize prediction matrix and counters
    processed_prediction_matrix = np.zeros((num_documents, num_topics))
    processed_data_topic_counts = np.zeros(num_topics, dtype=int)
    failed_predictions = []

    print(f"Predicting topics for {len(processed_data)} pre-processed documents...")

    # Process each pre-processed document
    for idx, doc_tokens in enumerate(processed_data):
        # Skip empty documents
        if not doc_tokens or len(doc_tokens) == 0:
            failed_predictions.append(idx)
            print(f"Skipping document {idx}: empty tokens")
            continue

        # Create a simple text from tokens for prediction
        # Since this is already processed, we join tokens back to text
        if isinstance(doc_tokens, list):
            text = " ".join(doc_tokens)
        else:
            text = str(doc_tokens)

        # Get prediction using pre-loaded components
        prediction_result = predict_topics_with_components(text, W, H, vocab, top_n=num_topics)

        if prediction_result is None:
            failed_predictions.append(idx)
            print(f"Failed to predict topics for document {idx}")
            continue

        # Extract L1 normalized scores and fill the matrix row
        l1_scores = prediction_result['l1_scores']
        for topic_name, score in l1_scores:
            topic_idx = int(topic_name.split()[1]) - 1  # Convert "Topic 1" to index 0
            processed_prediction_matrix[idx, topic_idx] = score

        # Count the highest scoring topic for this document
        highest_topic_name, _ = l1_scores[0]  # First item is highest scored
        highest_topic_idx = int(highest_topic_name.split()[1]) - 1
        processed_data_topic_counts[highest_topic_idx] += 1

        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(processed_data)} pre-processed documents...")

    document_indices = list(range(num_documents))

    print(f"Pre-processed data prediction complete. Successfully processed {len(processed_data) - len(failed_predictions)} documents.")
    print(f"Failed predictions: {len(failed_predictions)} documents")

    return {
        'processed_prediction_matrix': processed_prediction_matrix,
        'topic_labels': topic_labels,
        'document_indices': document_indices,
        'processed_data_topic_counts': processed_data_topic_counts,
        'failed_predictions': failed_predictions
    }


def predict_topics_for_dataset(dataset_path, column_name, model_file="topic_model_components.npz", separator=","):
    """
    Predict topics for an entire dataset and return a prediction matrix.

    Args:
        dataset_path (str): Path to the CSV dataset file
        column_name (str): Name of the column containing text data
        model_file (str): Path to saved model components
        separator (str): CSV separator (default: ",")

    Returns:
        dict: Dictionary containing:
            - prediction_matrix: numpy array where rows are documents, columns are topics
            - topic_labels: list of topic labels (e.g., ["Topic 1", "Topic 2", ...])
            - document_indices: list of document row indices from the dataset
            - failed_predictions: list of indices where prediction failed
    """
    # Load model components once at the beginning
    W, H, vocab = load_model_components(model_file)
    if H is None:
        print("Failed to load model components")
        return None

    num_topics = H.shape[0]
    topic_labels = [f"Topic {i+1}" for i in range(num_topics)]

    # Load dataset
    try:
        df = pd.read_csv(dataset_path, sep=separator)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in dataset. Available columns: {list(df.columns)}")
        return None

    # Initialize prediction matrix
    num_documents = len(df)
    prediction_matrix = np.zeros((num_documents, num_topics))
    failed_predictions = []

    print(f"Predicting topics for {num_documents} documents...")

    # Process each document using the optimized function
    for idx, row in df.iterrows():
        text = row[column_name]

        # Skip empty or non-string texts
        if pd.isna(text) or not hasattr(text, 'strip') or not text.strip():
            failed_predictions.append(idx)
            print(f"Skipping document {idx}: invalid text")
            continue

        # Get prediction for this document using pre-loaded components
        prediction_result = predict_topics_with_components(text, W, H, vocab, top_n=num_topics)

        if prediction_result is None:
            failed_predictions.append(idx)
            print(f"Failed to predict topics for document {idx}")
            continue

        # Extract L1 normalized scores and fill the matrix row
        l1_scores = prediction_result['l1_scores']
        for topic_name, score in l1_scores:
            topic_idx = int(topic_name.split()[1]) - 1  # Convert "Topic 1" to index 0
            prediction_matrix[idx, topic_idx] = score

        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{num_documents} documents...")

    document_indices = list(range(num_documents))

    print(f"Prediction complete. Successfully processed {num_documents - len(failed_predictions)} documents.")
    print(f"Failed predictions: {len(failed_predictions)} documents")

    # Calculate document counts per topic from prediction matrix
    # For each document, find the topic with highest prediction score
    prediction_topic_counts = np.zeros(num_topics, dtype=int)
    successful_docs = [i for i in range(num_documents) if i not in failed_predictions]

    for doc_idx in successful_docs:
        highest_topic_idx = np.argmax(prediction_matrix[doc_idx, :])
        prediction_topic_counts[highest_topic_idx] += 1

    # Calculate document counts per topic from W matrix
    # W matrix: documents (rows) × topics (columns)
    # For each document (row in W), find the topic with highest weight
    print(f"W matrix shape: {W.shape}, H matrix shape: {H.shape}")
    print(f"W matrix: {W.shape[0]} documents × {W.shape[1]} topics")
    print(f"H matrix: {H.shape[0]} topics × {H.shape[1]} words")

    # Count documents assigned to each topic based on W matrix
    w_topic_counts = np.zeros(num_topics, dtype=int)

    # Only count documents that were successfully processed
    for doc_idx in successful_docs:
        if doc_idx < W.shape[0]:  # Make sure document index is valid in W matrix
            highest_topic_idx = np.argmax(W[doc_idx, :])  # Topic with highest weight for this document
            w_topic_counts[highest_topic_idx] += 1

    return {
        'prediction_matrix': prediction_matrix,
        'topic_labels': topic_labels,
        'document_indices': document_indices,
        'failed_predictions': failed_predictions,
        'prediction_topic_counts': prediction_topic_counts,
        'w_matrix_topic_counts': w_topic_counts
    }


if __name__ == '__main__':
    file_path = "../veri_setleri/bbc_news.csv"
    column = "text"

    result = manta.run_topic_analysis(
        filepath=file_path,
        column=column,
        separator=",",
        language="EN",
        lemmatize=True,
        topic_count=10,
        words_per_topic=15,
        nmf_method="nmf", # "nmf" or "nmtf" or "pnmf"
        tokenizer_type="bpe",
        filter_app=True,
        data_filter_options = {
            "filter_app_country": "TR",
            "filter_app_country_column": "REVIEWER_LANGUAGE",
        },
        emoji_map=True,
        generate_wordclouds=True,
        save_to_db=False,
        word_pairs_out=False,
        topic_distribution=True,
    )

    old_str = """
    Manchester City secured a crucial 2-1 victory over Arsenal at the Emirates Stadium, extending their lead at the top of the Premier League table to five points. Erling Haaland opened the scoring in the 23rd minute with a trademark finish, capitalizing on a perfectly weighted through ball from Kevin De Bruyne.
     claim their third consecutive Premier League title. Ayrıca bu kazanç, ingiltere liginde ilgiye sebep oldu
     """
    test_text ="""
    COVID-19 pandemic has underscored the need for a well-trained public health workforce to save lives through timely outbreaks detection and response  In Yemen  a country that is entering its seventh year of a protracted war  the ongoing conflict severely limited the country s capacity to implement effective preparedness and response measures to outbreaks including COVID-19  There are growing concerns that the virus may be circulating within communities undetected and unmitigated especially as underreporting continues in some areas of the country due to a lack of testing facilities  delays in seeking treatment  stigma  difficulty accessing treatment centers  the perceived risks of seeking care or for political issues  The Yemen Field Epidemiology Training Program  FETP  was launched in 2011 to address the shortage of a skilled public health workforce  with the objective of strengthening capacity in field epidemiology  Thus  events of public health importance can be detected and investigated in a timely and effective manner  During the COVID-19 pandemic  the Yemen FETP s response has been instrumental through participating in country-level coordination  planning  monitoring  and developing guidelines standard operating procedures and strengthening surveillance capacities  outbreak investigations  contact tracing  case management  infection prevention  and control  risk communication  and research  As the third wave is circulating with a steeper upward curve than the previous ones with possible new variants  the country will not be able to deal with a surge of cases as secondary care is extremely crippled  Since COVID-19 prevention and control are the only option available to reduce its grave impact on morbidity and mortality  health partners should support the Yemen FETP to strengthen the health system s response to future epidemics  One important lesson learned from the COVID-19 pandemic  especially in the Yemen context and applicable to developing and war-torn countries  is that access to outside experts becomes limited  therefore  it is crucial to invest in building national expertise to provide timely  cost-effective  and sustainable services that are culturally appropriate  It is also essential to build such expertise at the governorate and district levels  as they are normally the first respondents  and to provide them with the necessary tools for immediate response in order to overcome the disastrous delays
    """

    model_file = result.get("model_file", None)

    print("\n=== Testing save/load workflow ===")
    # Test the new prediction function with the same text
    test_predictions = predict_topics(test_text, model_file=model_file, top_n=5)
    if test_predictions:
        print("Top predictions using saved model:")
        for i in range(len(test_predictions['raw_scores'])):
            raw_topic, raw_score = test_predictions['raw_scores'][i]
            l1_topic, l1_score = test_predictions['l1_scores'][i]
            print(f"{raw_topic} raw: {raw_score:.4f} L1: {l1_score:.4f}")

    print("\n=== Testing dataset-wide prediction matrix ===")
    # Predict topics for the entire dataset
    dataset_predictions = predict_topics_for_dataset(
        dataset_path=file_path,
        column_name=column,
        model_file=model_file,
        separator=","
    )

    print("\n=== Testing pre-processed data prediction ===")
    # Get processed data from the manta result
    processed_data = result.get("processed_data", None)
    if processed_data and dataset_predictions:
        # Load model components for processed data prediction
        W, H, vocab = load_model_components(model_file)
        if H is not None:
            processed_predictions = predict_topics_for_processed_data(processed_data, W, H, vocab)
        else:
            processed_predictions = None
    else:
        processed_predictions = None

    if dataset_predictions:
        prediction_matrix = dataset_predictions['prediction_matrix']
        topic_labels = dataset_predictions['topic_labels']
        failed_predictions = dataset_predictions['failed_predictions']
        prediction_topic_counts = dataset_predictions['prediction_topic_counts']
        w_matrix_topic_counts = dataset_predictions['w_matrix_topic_counts']

        print(f"Raw data prediction matrix shape: {prediction_matrix.shape}")
        print(f"Topics: {topic_labels}")
        print(f"Raw data failed predictions: {len(failed_predictions)} out of {prediction_matrix.shape[0]} documents")

        if processed_predictions:
            processed_prediction_matrix = processed_predictions['processed_prediction_matrix']
            processed_failed = processed_predictions['failed_predictions']
            print(f"Processed data prediction matrix shape: {processed_prediction_matrix.shape}")
            print(f"Processed data failed predictions: {len(processed_failed)} out of {processed_prediction_matrix.shape[0]} documents")

        # Show document counts per topic comparison
        print("\n=== Topic Document Counts Comparison ===")
        if processed_predictions:
            processed_topic_counts = processed_predictions['processed_data_topic_counts']
            print(f"{'Topic':<12} {'Raw Data':<10} {'W Matrix':<10} {'Processed Data':<15} {'Raw-W':<8} {'Raw-Proc':<10}")
            print("-" * 75)
            total_pred_docs = 0
            total_w_docs = 0
            total_proc_docs = 0
            for i, topic_label in enumerate(topic_labels):
                pred_count = prediction_topic_counts[i]
                w_count = w_matrix_topic_counts[i]
                proc_count = processed_topic_counts[i]
                diff_w = pred_count - w_count
                diff_proc = pred_count - proc_count
                print(f"{topic_label:<12} {pred_count:<10} {w_count:<10} {proc_count:<15} {diff_w:+8d} {diff_proc:+10d}")
                total_pred_docs += pred_count
                total_w_docs += w_count
                total_proc_docs += proc_count

            print("-" * 75)
            print(f"{'TOTAL':<12} {total_pred_docs:<10} {total_w_docs:<10} {total_proc_docs:<15} {total_pred_docs - total_w_docs:+8d} {total_pred_docs - total_proc_docs:+10d}")

            print(f"\nExplanation:")
            print(f"- Raw Data: Documents assigned to each topic (from CSV, with preprocessing)")
            print(f"- W Matrix: Documents assigned to each topic (from original NMF training)")
            print(f"- Processed Data: Documents assigned to each topic (from manta pre-processed data)")
            print(f"- Raw-W: Difference between raw data predictions and W matrix assignments")
            print(f"- Raw-Proc: Difference between raw data and processed data predictions")
        else:
            print(f"{'Topic':<12} {'Prediction Matrix':<18} {'W Matrix':<10} {'Difference':<12}")
            print("-" * 55)
            total_pred_docs = 0
            total_w_words = 0
            for i, topic_label in enumerate(topic_labels):
                pred_count = prediction_topic_counts[i]
                w_count = w_matrix_topic_counts[i]
                diff = pred_count - w_count
                print(f"{topic_label:<12} {pred_count:<18} {w_count:<10} {diff:+12d}")
                total_pred_docs += pred_count
                total_w_words += w_count

            print("-" * 55)
            print(f"{'TOTAL':<12} {total_pred_docs:<18} {total_w_words:<10} {total_pred_docs - total_w_words:+12d}")

            print(f"\nExplanation:")
            print(f"- Prediction Matrix counts: Documents assigned to each topic (based on highest prediction score)")
            print(f"- W Matrix counts: Documents assigned to each topic (based on highest topic weight in training)")
            print(f"- Total prediction documents: {total_pred_docs} (should equal successful predictions)")
            print(f"- Total W matrix documents: {total_w_words} (should also equal successful predictions)")
            print(f"- Differences show how prediction results differ from original NMF training assignments")

        # Show some example predictions (first 5 documents)
        print("\nExample predictions (first 5 documents):")
        for i in range(min(5, prediction_matrix.shape[0])):
            if i not in failed_predictions:
                print(f"Document {i}:")
                # Show top 3 topics for this document
                doc_scores = [(topic_labels[j], prediction_matrix[i, j]) for j in range(len(topic_labels))]
                doc_scores_sorted = sorted(doc_scores, key=lambda x: x[1], reverse=True)
                for topic, score in doc_scores_sorted[:3]:
                    print(f"  {topic}: {score:.4f}")
            else:
                print(f"Document {i}: Failed to predict")

        # Save the matrices with additional data
        matrix_file = "prediction_matrix.npz"
        save_data = {
            'raw_prediction_matrix': prediction_matrix,
            'topic_labels': topic_labels,
            'raw_failed_predictions': failed_predictions,
            'prediction_topic_counts': prediction_topic_counts,
            'w_matrix_topic_counts': w_matrix_topic_counts
        }

        # Add processed data matrix and counts if available
        if processed_predictions:
            save_data['processed_prediction_matrix'] = processed_predictions['processed_prediction_matrix']
            save_data['processed_data_topic_counts'] = processed_predictions['processed_data_topic_counts']
            save_data['processed_data_failed_predictions'] = processed_predictions['failed_predictions']

        np.savez(matrix_file, **save_data)
        print(f"\nPrediction matrices and topic counts saved to {matrix_file}")

        # Show matrix information
        print(f"\nSaved data includes:")
        print(f"- raw_prediction_matrix: {prediction_matrix.shape}")
        if processed_predictions:
            print(f"- processed_prediction_matrix: {processed_predictions['processed_prediction_matrix'].shape}")
        print(f"- topic_labels: {len(topic_labels)} topics")
        print(f"- All topic count comparisons and failed prediction indices")