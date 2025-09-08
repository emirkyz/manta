import numpy as np

from ...utils.analysis.distance_two_words import calc_levenstein_distance, calc_cosine_distance
from ...utils.database.database_manager import DatabaseManager


def _sort_matrices(s: np.ndarray) -> tuple[list[tuple[int, int]], list[float]]:
    ind = []
    max_values = []

    for i in range(s.shape[1]):
        col = s[:, i]
        max_ind = np.argmax(col)
        max_values.append(col[max_ind])
        ind.append((i, max_ind))

    ind_sorted = np.argsort(max_values)[::-1]
    ind = np.array(ind)[ind_sorted]
    max_values = np.array(max_values)[ind_sorted]

    return ind, max_values


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


def topic_extract(H, W, topic_count, tokenizer=None, vocab=None, documents=None, db_config=None, data_frame_name=None, word_per_topic=20, include_documents=True, emoji_map=None, doc_word_pairs=None):
    """
    Performs topic analysis using Non-negative Matrix Factorization (NMF) results for both Turkish and English texts.
    
    This function extracts meaningful topics from NMF decomposition matrices by identifying the most 
    significant words for each topic and optionally analyzing the most relevant documents. It supports
    both Turkish (using tokenizer) and English (using vocabulary list) processing pipelines.
    
    Args:
        H (numpy.ndarray): Topic-word matrix from NMF decomposition with shape (n_topics, n_features).
                          Each row represents a topic, each column represents a word/feature.
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
        doc_word_pairs (list[tuple[int, int]], optional): List of (word_topic_id, doc_topic_id) pairs for NMTF-style analysis.
                                                         If provided, only these specific topic pairs will be analyzed.
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


    # Handle NMTF-style doc_word_pairs or standard topic iteration
    if doc_word_pairs is not None:
        # doc_word_pairs should be a list of (word_topic_id, doc_topic_id) tuples for NMTF
        """
        These matrices are expected to be in NMTF format:
        W : Document-topic matrix (n_documents, n_topics)
        H : Topic-word matrix (n_topics, n_features)
        doc_word_pairs : List of (word_topic_id, doc_topic_id) tuples for NMTF-style analysis
        """
        ind, max_vals = _sort_matrices(doc_word_pairs)
        for idx, (word_vec_id, doc_vec_id) in enumerate(ind):
            if topic_count == -1:
                topic_count = H.shape[0]
            # Extract topic vectors for this specific pair
            # Handle both sparse and dense matrices
            topic_word_vector = H[word_vec_id, :]
            topic_doc_vector = W[:, doc_vec_id]

            # Get sorted indices by score (highest first)
            sorted_word_ids = np.flip(np.argsort(topic_word_vector))
            sorted_doc_ids = np.flip(np.argsort(topic_doc_vector))

            # Extract words for this topic pair
            word_scores = _extract_topic_words(
                topic_word_vector, sorted_word_ids, tokenizer, vocab, emoji_map, word_per_topic
            )
            word_result[f"Topic {idx+1:02d}"] = word_scores
            
            # Extract documents for this topic pair (optional)
            if include_documents and documents is not None:
                top_doc_ids = sorted_doc_ids[:10]  # TODO: make configurable
                doc_scores = _extract_topic_documents(
                    topic_doc_vector, top_doc_ids, documents, emoji_map
                )
                document_result[f"Topic {idx+1}"] = doc_scores
    else:
        if topic_count==-1:
            topic_count = H.shape[0]
        # Standard NMF: Process all topics sequentially
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
