import os
import numpy as np
import scipy.sparse as sp
from ..functions.nmf.nmf_orchestrator import run_nmf
from ..functions.common_language.topic_analyzer import konu_analizi
from .save_doc_score_pair import save_doc_score_pair
from .coherence_score import calculate_coherence_scores


def hierarchy_nmf(W, nmf_matrix, selected_topic, desired_topic_count,
                  nmf_method, sozluk, tokenizer, metin_array,
                  topics_db_eng, table_name, emoji_map,
                  word_per_topic=15, base_dir=None, output_dir=None):
    """
    Hierarchy NMF - Performs hierarchical topic modeling on a selected topic.
    
    Args:
        W: Document-topic matrix from initial NMF
        nmf_matrix: Term-document matrix (sparse matrix)
        selected_topic: Index of the topic to analyze hierarchically
        desired_topic_count: Number of sub-topics to extract
        nmf_method: NMF algorithm method to use
        sozluk: Vocabulary/dictionary
        tokenizer: Tokenizer instance
        metin_array: Array of cleaned text documents
        topics_db_eng: Database engine for topics
        table_name: Name for the analysis table
        emoji_map: Emoji mapping instance
        word_per_topic: Number of words per topic (default: 15)
        base_dir: Base directory path
        output_dir: Output directory path
    
    Returns:
        result: Topic analysis results
    """
    selected_topic = 3
    desired_topic_count = 3



    # Create hierarchy output directory
    hierarchy_output_dir = os.path.join(output_dir, table_name + "_hierarchy")
    os.makedirs(hierarchy_output_dir, exist_ok=True)
    
    # Find non-zero indexes from the selected topic in W matrix
    print(f"Size of tdm: {nmf_matrix.shape}")
    #Get non zero ones
    non_zero_indexes = np.where(W[:, selected_topic] >= 0.01)[0]
    non_zero_index_values = W[non_zero_indexes, selected_topic].reshape(-1, 1)
    
    
    #Get max scored ones
    #non_zero_indexes = np.argmax(W[:, selected_topic],axis=1)
    # Get the non-zero rows from the tdm sparse matrix
    nmf_matrix_selected_topic = nmf_matrix[non_zero_indexes]
    h_tdm = nmf_matrix_selected_topic
    print(f"Size of tdm after filtering: {h_tdm.shape}")

    h_multiply = False
    if h_multiply:
        # Scale each row of h_tdm with corresponding non_zero_index_values using matrix multiplication
        # Create diagonal matrix from the scaling values
        scaling_diagonal = sp.diags(non_zero_index_values.flatten(), format='csr')
        h_tdm = scaling_diagonal @ h_tdm

    # Filter documents for the selected topic
    filtered_metin_array = [metin_array[i] for i in non_zero_indexes]
    
    print("\n")
    print("--------------------------------")
    print("--------------------------------")
    print(f"Hierarchy NMF: {table_name} - Selected Topic: {selected_topic} - Desired Topic Count: {desired_topic_count}")
    print(f"Length of filtered_metin_array: {len(filtered_metin_array)}")

    counnt_of_nonzero = h_tdm.count_nonzero()
    print(f"count nonzero : ", counnt_of_nonzero)
    total_elements = h_tdm.shape[0] * h_tdm.shape[1]
    print(f"total elements : ", total_elements)
    max_optimal_topic_num = counnt_of_nonzero // (desired_topic_count + len(sozluk))
    print("max_optimal_topic_num : ", max_optimal_topic_num)
    
    
    topics_numbers_to_test = [2,3,4,5]
    for topics_number in topics_numbers_to_test:
        # Create subfolder for this specific topic number
        topic_output_dir = os.path.join(hierarchy_output_dir, str(topics_number))
        os.makedirs(topic_output_dir, exist_ok=True)
        
        W_hierarchy, H_hierarchy = run_nmf(
            num_of_topics=int(topics_number),
            sparse_matrix=h_tdm,
            norm_thresh=0.005,
            nmf_method=nmf_method
        )
        result = konu_analizi(
                H=H_hierarchy,
                W=W_hierarchy,
                konu_sayisi=int(topics_number),
                sozluk=sozluk,
                tokenizer=tokenizer,
                documents=filtered_metin_array,
                topics_db_eng=topics_db_eng,
                data_frame_name=table_name + "_hierarchy_" + str(topics_number),
                word_per_topic=word_per_topic,
                include_documents=True,
                emoji_map=emoji_map,
                output_dir=topic_output_dir
            )
        print(f"Topics number: {topics_number} - ")
        topic_word_scores = save_doc_score_pair(
                            base_dir,
                            topic_output_dir,
                            table_name + "_hierarchy",
                            result,
                            H_hierarchy
                            )
        coherence_scores = calculate_coherence_scores(topic_word_scores, topic_output_dir,cleaned_data=filtered_metin_array,table_name=f"hierarchy_{topics_number}_topics")
        #gen_topic_dist(W_hierarchy, topic_output_dir, table_name + "_hierarchy_" + str(topics_number))
        #top_pairs = calc_word_cooccurrence(H_hierarchy, sozluk, topic_output_dir, table_name+"_hierarchy_" + str(topics_number), top_n=50, min_score=5,
        #                                language="TR", tokenizer=tokenizer,create_heatmap=True)

    return None