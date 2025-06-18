import json
import os

import numpy as np

from functions.english.topics import save_topics_to_db
from utils.distance_two_words import calc_levenstein_distance, calc_cosine_distance


def konu_analizi(H, W, konu_sayisi, tokenizer=None, sozluk=None, documents=None, topics_db_eng=None, data_frame_name=None, word_per_topic=20, include_documents=True):
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
        konu_sayisi (int): Number of topics to analyze. Should match the number of topics in H and W matrices.
        tokenizer (object, optional): Turkish tokenizer object with id_to_token() method for converting 
                                    token IDs to words. Required for Turkish text processing.
        sozluk (list, optional): English vocabulary list where indices correspond to feature indices in H matrix.
                               Required for English text processing.
        documents (pandas.DataFrame or list, optional): Collection of document texts used in the analysis.
                                                       Can be pandas DataFrame or list of strings.
        topics_db_eng (sqlalchemy.engine, optional): Database engine for saving topic results to database.
        data_frame_name (str, optional): Name of the dataset/table, used for file naming and database operations.
        word_per_topic (int, optional): Maximum number of top words to extract per topic. Default is 20.
        include_documents (bool, optional): Whether to perform document analysis and save document scores.
                                          Default is True.
    
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
    if tokenizer is None and sozluk is None:
        raise ValueError("Either tokenizer (for Turkish) or sozluk (for English) must be provided")
    
    result = {}
    dokuman_result = {}
    
    for i in range(konu_sayisi):
        konu_kelime_vektoru = H[i, :]
        konu_dokuman_vektoru = W[:, i]

        sirali_kelimeler = np.flip(np.argsort(konu_kelime_vektoru))
        sirali_haberler = np.flip(np.argsort(konu_dokuman_vektoru))

        ilk_kelimeler = sirali_kelimeler
        ilk_10_dokuman = sirali_haberler[:20]

        # Get the words and their corresponding scores in "word:score" format
        kelime_skor_listesi = []
        for id in ilk_kelimeler:
            # Get word based on whether we're using tokenizer (Turkish) or sozluk (English)
            if tokenizer is not None:
                kelime = tokenizer.id_to_token(id)
            else:  # Using sozluk for English
                if id < len(sozluk):
                    kelime = sozluk[id]
                else:
                    continue

            # Skip subword tokens that start with ##
            if kelime is not None and kelime.startswith("##"):
                continue
            
            # calculate distance between previous word and current word
            # if distance is less than 3, skip the current word
            if id > 0 and kelime_skor_listesi:  # Only check distance if we have previous words
                # check for all previous words
                for prev_word in kelime_skor_listesi:
                    prev_word_org = prev_word.split(":")[0]
                    prev_word_text = prev_word_org
                    if "/" in prev_word_text:
                        # If the previous word is already combined, use the first part
                        prev_word_text = prev_word_text.split("/")[0].strip()

                    distance = calc_cosine_distance(prev_word_text, kelime)
                    leven_distance = calc_levenstein_distance(prev_word_text, kelime)
                    if distance > 0.8:
                        # combine two words
                        kelime = f"{prev_word_org} / {kelime}"
                        kelime_skor_listesi.remove(prev_word)

                    #elif leven_distance < 3 and len(kelime) != leven_distance:
                    #    kelime = f"{prev_word_text} / {kelime}"
                    #    kelime_skor_listesi.remove(prev_word)

            # Add word and score to the list
            
            skor = konu_kelime_vektoru[id]
            kelime_skor_listesi.append(f"{kelime}:{skor:.8f}")
            if len(kelime_skor_listesi) >= word_per_topic:
                break

        result[f"Konu {i:02d}"] = kelime_skor_listesi
        
        # Document analysis (optional)
        if include_documents and documents is not None:
            document_skor_listesi = {}
            for id in ilk_10_dokuman:
                if id < len(documents):
                    skor = konu_dokuman_vektoru[id]
                    document_skor_listesi[f"{id}"] = f"{documents.iloc[id] if hasattr(documents, 'iloc') else documents[id]}:{skor:.4f}"
            dokuman_result[f"Konu {i}"] = document_skor_listesi

    # Save document analysis if it was generated
    if include_documents and documents is not None and data_frame_name:
        # Create table-specific subdirectory under Output folder  
        base_dir = os.path.abspath(os.path.dirname(__file__))
        # Go up two levels to get to the project root, then into Output
        output_dir = os.path.join(base_dir, "..", "..", "Output")
        table_output_dir = os.path.join(output_dir, data_frame_name)
        os.makedirs(table_output_dir, exist_ok=True)
        
        # Save document scores to table-specific subdirectory
        document_file_path = os.path.join(table_output_dir, f"top_docs_{data_frame_name}.json")
        json.dump(dokuman_result, open(document_file_path, "w"), ensure_ascii=False)

    # Save topics to database
    if topics_db_eng:
        save_topics_to_db(result, data_frame_name, topics_db_eng)
    else:
        print("Warning: No database engine provided, skipping database save")
        
    return result
