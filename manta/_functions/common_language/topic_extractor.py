import numpy as np

from ...utils.analysis.distance_two_words import calc_levenstein_distance, calc_cosine_distance
from ...utils.database.database_manager import DatabaseManager


def _get_word_cluster_for_doc_cluster(s_matrix: np.ndarray, doc_cluster_idx: int) -> int:
    """
    For a given doc-cluster (W column), find the best matching word-cluster (H row).

    S matrix structure: S[j, i] = coupling between W column i and H row j
    - Column i corresponds to doc-cluster i (W[:, i])
    - Row j corresponds to word-cluster j (H[j, :])

    Args:
        s_matrix: S matrix (k x k) where S[j, i] = coupling between W[:,i] and H[j,:]
        doc_cluster_idx: Index of the document cluster (W column)

    Returns:
        Index of the best matching word cluster (H row)
    """
    # Find word-cluster (row j) with maximum coupling to this doc-cluster (column i)
    return np.argmax(s_matrix[doc_cluster_idx, :])


def _process_word_token(word_id, tokenizer, vocabulary, emoji_map):
    """
    Process a single word token, handling tokenizer vs sozluk and emoji decoding.
    
    Args:
        word_id (int): Token ID to process  
        tokenizer: Turkish tokenizer object (optional)
        vocabulary (list): English vocabulary list (optional)
        emoji_map: Emoji map for decoding (optional)
        
    Returns:
        str or None: Processed word token, or None if invalid/filtered
    """
    if tokenizer is not None:
        word = tokenizer.id_to_token(word_id)
    else:
        if word_id < len(vocabulary):
            word = vocabulary[word_id]
        else:
            return None
    
    # Handle emoji decoding
    if emoji_map is not None and word is not None:
        if emoji_map.check_if_text_contains_tokenized_emoji(word):
            word = emoji_map.decode_text(word)
    
    # Skip subword tokens that start with ##
    if word is not None and word.startswith("##"):
        return None
        
    return word


def _apply_word_similarity_filtering(word, word_score_list):
    """
    Apply similarity filtering to combine similar words.
    
    Args:
        word (str): Current word to check
        word_score_list (list): List of existing word:score strings
        
    Returns:
        tuple: (processed_word, updated_list) where processed_word might be combined
    """
    if not word_score_list:
        return word, word_score_list
        
    for prev_word in word_score_list[:]:
        prev_word_org = prev_word.split(":")[0]
        prev_word_text = prev_word_org
        if "/" in prev_word_text:
            prev_word_text = prev_word_text.split("/")[0].strip()

        distance = calc_levenstein_distance(prev_word_text, word)
        distance = calc_cosine_distance(prev_word_text, word)
        if distance > 0.8:
            word = f"{prev_word_org} / {word}"
            word_score_list.remove(prev_word)
            break
            
    return word, word_score_list


def _extract_topic_words(topic_word_vector, word_ids, tokenizer, vocabulary, emoji_map, word_per_topic):
    """
    Extract and process words for a single topic.
    
    Args:
        topic_word_vector (numpy.ndarray): Word scores for this topic
        word_ids (numpy.ndarray): Sorted word IDs by score
        tokenizer: Turkish tokenizer (optional)
        vocabulary (list): English vocabulary (optional)
        emoji_map: Emoji map for decoding (optional)
        word_per_topic (int): Maximum words per topic
        
    Returns:
        list: List of word:score strings
    """
    word_score_list = []
    
    for word_id in word_ids:
        word = _process_word_token(word_id, tokenizer, vocabulary, emoji_map)
        if word is None:
            continue
            
        word, word_score_list = _apply_word_similarity_filtering(word, word_score_list)
        
        score = topic_word_vector[word_id]
        word_score_list.append(f"{word}:{score:.8f}")
        
        if len(word_score_list) >= word_per_topic:
            break
            
    return word_score_list


def _extract_topic_documents(topic_doc_vector, doc_ids, documents, emoji_map):
    """
    Extract and process documents for a single topic.

    Args:
        topic_doc_vector (numpy.ndarray): Document scores for this topic
        doc_ids (numpy.ndarray): Sorted document IDs by score
        documents: Collection of documents (DataFrame or list)
        emoji_map: Emoji map for decoding (optional)

    Returns:
        dict: Dictionary of document_id -> document_text:score strings
    """
    document_score_list = {}

    for doc_id in doc_ids:
        if doc_id < len(documents):
            score = topic_doc_vector[doc_id]

            # Skip documents with zero or negative scores
            if score <= 0.0:
                continue

            if hasattr(documents, 'iloc'):
                document_text = documents.iloc[doc_id]
            else:
                document_text = documents[doc_id]

            if emoji_map is not None:
                if emoji_map.check_if_text_contains_tokenized_emoji_doc(document_text):
                    document_text = emoji_map.decode_text_doc(document_text)
            document_text = document_text.translate(str.maketrans('', '', '\n')).replace('\"', '')
            document_score_list[f"{doc_id}"] = f"{document_text}:{score:.16f}"

    return document_score_list


def topic_extract(H, W, topic_count, tokenizer=None, vocab=None, documents=None, db_config=None, data_frame_name=None, word_per_topic=20, include_documents=True, emoji_map=None, s_matrix=None):
    """
    Performs topic analysis using Non-negative Matrix Factorization (NMF) results for both Turkish and English texts.

    This function extracts meaningful topics from NMF decomposition matrices by identifying the most
    significant words for each topic and optionally analyzing the most relevant documents. It supports
    both Turkish (using tokenizer) and English (using vocabulary list) processing pipelines.

    For NMTF (Non-negative Matrix Tri-Factorization), this function uses the NMF-Equivalent Method where
    document clusters are treated as primary topics and words are projected onto the document-cluster space
    via the transformation H' = S @ H.

    Args:
        H (numpy.ndarray): Topic-word matrix from NMF decomposition with shape (n_topics, n_features).
                          Each row represents a topic, each column represents a word/feature.
                          For NMTF, this should be the original H matrix before S transformation.
        W (numpy.ndarray): Document-topic matrix from NMF decomposition with shape (n_documents, n_topics).
                          Each row represents a document, each column represents a topic.
        topic_count (int): Number of topics to analyze. Should match the number of topics in H and W matrices.
        tokenizer (object, optional): Turkish tokenizer object with id_to_token() method for converting
                                    token IDs to words. Required for Turkish text processing.
        vocab (list, optional): English vocabulary list where indices correspond to feature indices in H matrix.
                               Required for English text processing.
        documents (pandas.DataFrame or list, optional): Collection of document texts used in the analysis.
                                                       Can be pandas DataFrame or list of strings.
        db_config (DatabaseConfig, optional): Database configuration object containing database engines and output directories.
        data_frame_name (str, optional): Name of the dataset/table, used for database operations and file naming.
        word_per_topic (int, optional): Maximum number of top words to extract per topic. Default is 20.
        include_documents (bool, optional): Whether to perform document analysis and save document scores.
                                          Default is True.
        emoji_map (EmojiMap, optional): Emoji map for decoding emoji tokens back to emojis. Required for Turkish text processing.
        s_matrix (numpy.ndarray, optional): Core matrix from NMTF decomposition with shape (n_topics, n_topics).
                                           If provided, H will be transformed to H' = S @ H to project words onto document clusters.
    Returns:
        dict: Dictionary where keys are topic names in format "Konu XX" and values are lists of 
              word-score strings in format "word:score". Scores are formatted to 8 decimal places.
              
    Raises:
        ValueError: If neither tokenizer (for Turkish) nor sozluk (for English) is provided.
        
    Side Effects:
        - Creates directory structure: {project_root}/Output/{data_frame_name}/ (if include_documents=True)
        - Saves JSON file: top_docs_{data_frame_name}.json with document analysis results
        - Saves topics to database if topics_db_eng is provided
        - Prints warning message if no database engine is provided
        
    Examples:
        # Turkish text analysis
        result = konu_analizi(
            H=topic_word_matrix,
            W=doc_topic_matrix, 
            konu_sayisi=5,
            tokenizer=turkish_tokenizer,
            documents=turkish_docs,
            data_frame_name="turkish_news"
        )
        
        # English text analysis  
        result = konu_analizi(
            H=topic_word_matrix,
            W=doc_topic_matrix,
            konu_sayisi=3,
            sozluk=english_vocab,
            documents=english_docs,
            topics_db_eng=db_engine,
            data_frame_name="english_articles"
        )
        
        # Result format:
        # {
        #     "Konu 00": ["machine:0.12345678", "learning:0.09876543", ...],
        #     "Konu 01": ["data:0.11111111", "science:0.08888888", ...],
        #     ...
        # }
    
    Note:
        - Subword tokens starting with "##" are automatically filtered out
        - Words are ranked by their topic scores in descending order
        - Document analysis extracts top 20 documents per topic when enabled
        - Function works with both pandas DataFrames and regular lists for documents
        - Database saving is optional and warnings are shown if engine is not provided
        - File paths are resolved relative to the function's location in the project structure
    """
    if tokenizer is None and vocab is None:
        raise ValueError("Either tokenizer (for Turkish) or vocab (for English) must be provided")

    word_result = {}
    document_result = {}

    if s_matrix is not None:
        # NMTF mode: use sequential doc-cluster indices as topics
        # Map each doc-cluster (W column) to its best word-cluster (H row) via S matrix
        # S[j, i] = coupling between W column i and H row j
        if topic_count == -1:
            topic_count = W.shape[1]

        for topic_idx in range(topic_count):
            # Find best word-cluster (H row j) for this doc-cluster (W column i)
            word_cluster_idx = _get_word_cluster_for_doc_cluster(s_matrix, topic_idx)

            topic_word_vector = H[word_cluster_idx, :]
            topic_doc_vector = W[:, topic_idx]

            # Get sorted indices by score (highest first)
            sorted_word_ids = np.flip(np.argsort(topic_word_vector))
            sorted_doc_ids = np.flip(np.argsort(topic_doc_vector))

            # Extract words for this topic
            word_scores = _extract_topic_words(
                topic_word_vector, sorted_word_ids, tokenizer, vocab, emoji_map, word_per_topic
            )
            word_result[f"Topic {topic_idx+1:02d}"] = word_scores

            # Extract documents for this topic (optional)
            if include_documents and documents is not None:
                top_doc_ids = sorted_doc_ids[:10]
                doc_scores = _extract_topic_documents(
                    topic_doc_vector, top_doc_ids, documents, emoji_map
                )
                document_result[f"Topic {topic_idx+1}"] = doc_scores
    else:
        # Standard NMF mode: iterate sequentially
        if topic_count == -1:
            topic_count = H.shape[0]

        for i in range(topic_count):
            topic_word_vector = H[i, :]
            topic_doc_vector = W[:, i]

            # Get sorted indices by score (highest first)
            sorted_word_ids = np.flip(np.argsort(topic_word_vector))
            sorted_doc_ids = np.flip(np.argsort(topic_doc_vector))

            # Extract words for this topic
            word_scores = _extract_topic_words(
                topic_word_vector, sorted_word_ids, tokenizer, vocab, emoji_map, word_per_topic
            )
            word_result[f"Topic {i+1:02d}"] = word_scores

            # Extract documents for this topic (optional)
            if include_documents and documents is not None:
                top_doc_ids = sorted_doc_ids[:10]
                doc_scores = _extract_topic_documents(
                    topic_doc_vector, top_doc_ids, documents, emoji_map
                )
                document_result[f"Topic {i+1}"] = doc_scores

    # Save to database if provided
    if db_config and db_config.topics_db_engine and data_frame_name:
        DatabaseManager.save_topics_to_database(word_result, data_frame_name, db_config.topics_db_engine)
    else:
        print("Warning: No database configuration or data frame name provided, skipping database save")
        
    return word_result, document_result
