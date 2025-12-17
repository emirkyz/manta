"""
Topic modeling pipeline for MANTA topic analysis.
"""

from typing import Dict, Any, Optional, Tuple

import pandas as pd

from .._functions.common_language.topic_extractor import topic_extract
from .._functions.nmf import run_nmf
from ..utils.analysis.coherence_score import calculate_coherence_scores
from ..utils.analysis.topic_correlation import build_correlation_graph
from ..utils.analysis.topic_similarity import HybridTFIDFTopicSimilarity
from ..utils.export.save_doc_score_pair import save_doc_score_pair
from ..utils.export.save_word_score_pair import save_word_score_pair
from ..utils.export.save_s_matrix import save_s_matrix
from ..utils.console.console_manager import ConsoleManager, get_console
from ..utils.visualization.topic_similarity_heatmap import plot_combined_similarity_analysis
import json
import numpy as np
from pathlib import Path


class ModelingPipeline:
    """Handles NMF topic modeling and analysis."""
    
    @staticmethod
    def perform_topic_modeling(
        tdm, 
        options: Dict[str, Any], 
        vocab, 
        text_array,
        db_config, 
        table_name: str, 
        table_output_dir, 
        console: Optional[ConsoleManager] = None,
        desired_columns: str = "text"
    ) -> Tuple[Dict, Dict, Dict, Dict, Any]:
        """
        Perform NMF topic modeling and analysis.
        
        Args:
            console: Console manager for status messages
        
        Returns:
            Tuple of (topic_word_scores, topic_doc_scores, coherence_scores, nmf_output, word_result)
        """
        _console = console or get_console()
        _console.print_status(f"Starting NMF processing ({options['nmf_type'].upper()})...", "processing")
        
        # NMF processing
        nmf_output = run_nmf(
            num_of_topics=int(options["DESIRED_TOPIC_COUNT"]),
            sparse_matrix=tdm,
            norm_thresh=0.005,
            nmf_method=options["nmf_type"],
        )

        _console.print_status("Extracting topics from NMF results...", "processing")
            
        # Extract topics based on language
        if options["LANGUAGE"] == "TR":
            word_result, document_result = topic_extract(
                H=nmf_output["H"],
                W=nmf_output["W"],
                s_matrix=nmf_output.get("S", None),
                topic_count=int(options["DESIRED_TOPIC_COUNT"]),
                vocab=vocab,
                tokenizer=options["tokenizer"],
                documents=text_array,
                db_config=db_config,
                data_frame_name=table_name,
                word_per_topic=options["N_TOPICS"],
                include_documents=True,
                emoji_map=options["emoji_map"],
            )
        elif options["LANGUAGE"] == "EN":
            word_result, document_result = topic_extract(
                H=nmf_output["H"],
                W=nmf_output["W"],
                s_matrix=nmf_output.get("S", None),
                topic_count=int(options["DESIRED_TOPIC_COUNT"]),
                vocab=vocab,
                documents=text_array,
                db_config=db_config,
                data_frame_name=table_name,
                word_per_topic=options["N_TOPICS"],
                include_documents=True,
                emoji_map=options["emoji_map"],
            )
        else:
            raise ValueError(f"Invalid language: {options['LANGUAGE']}")

        _console.print_status("Saving topic results...", "processing")
            
        # Convert the topics_data format to the desired format
        topic_word_scores = save_word_score_pair(
            base_dir=None,
            output_dir=table_output_dir,
            table_name=table_name,
            topics_data=word_result,
            result=None,
            data_frame_name=table_name,
            topics_db_eng=db_config.topics_db_engine,
        )
        
        # Save document result to json
        topic_doc_scores = save_doc_score_pair(
            document_result,
            base_dir=None,
            output_dir=table_output_dir,
            table_name=table_name,
            data_frame_name=table_name,
        )

        # Save S matrix if present (for NMTF)
        if "S" in nmf_output:
            _console.print_status("Saving S matrix...", "processing")
            save_s_matrix(
                s_matrix=nmf_output["S"],
                output_dir=table_output_dir,
                table_name=table_name,
                data_frame_name=table_name,
            )

            # Generate S matrix graph visualizations
            _console.print_status("Generating S matrix graph visualizations...", "processing")

            from ..utils.visualization.s_matrix_graph import visualize_s_matrix_graph
            visualize_s_matrix_graph(
                s_matrix=nmf_output["S"],
                output_dir=table_output_dir,
                table_name=table_name,
                threshold=0.01,
                layout="circular",
                create_interactive=False,
                create_heatmap=True
            )

        _console.print_status("Calculating coherence scores...", "processing")
            
        # Calculate and save coherence scores
        coherence_scores = calculate_coherence_scores(
            topic_word_scores,
            tokenizer=options["tokenizer"],
            output_dir=table_output_dir,
            column_name=desired_columns,
            cleaned_data=text_array,
            table_name=table_name,
            topic_word_matrix=nmf_output["H"],
            doc_topic_matrix=nmf_output["W"],
            vocabulary=vocab,
            s_matrix = nmf_output.get("S", None),
        )

        if False:
            # Calculate topic similarity using hybrid weighted TF-IDF
            _console.print_status("Computing topic similarity scores...", "processing")

            try:
                # Create vocabulary dict if it's a list
                if isinstance(vocab, list):
                    vocab_dict = {word: idx for idx, word in enumerate(vocab)}
                else:
                    vocab_dict = vocab

                # Get topic names from word_result
                topic_names = list(word_result.keys())

                # Initialize similarity scorer
                similarity_scorer = HybridTFIDFTopicSimilarity(
                    H_matrix=nmf_output["H"],
                    vocabulary=vocab_dict,
                    tfidf_matrix=tdm,  # Use the TF-IDF matrix to compute IDF values
                    topic_names=topic_names
                )

                # Compute weighted TF-IDF vectors and similarity matrix
                similarity_scorer.create_weighted_tfidf_vectors(
                    top_n_words=100,  # Focus on top 100 words per topic
                    normalize=True
                )

                similarity_matrix = similarity_scorer.compute_similarity_matrix(
                    method='cosine'
                )

                # Get summary statistics
                similarity_stats = similarity_scorer.get_summary_statistics()

                # Find redundant topics
                redundant_pairs = similarity_scorer.find_redundant_topics(
                    threshold=0.8
                )

                # Get merge suggestions
                merge_suggestions = similarity_scorer.suggest_topic_merging(
                    threshold=0.8,
                    method='hierarchical'
                )

                # Save results to JSON
                similarity_results = {
                    'n_topics': int(similarity_scorer.n_topics),
                    'topic_names': topic_names,
                    'similarity_matrix': similarity_matrix.tolist(),
                    'summary_statistics': similarity_stats,
                    'redundant_pairs': redundant_pairs,
                    'merge_suggestions': merge_suggestions
                }

                output_file = Path(table_output_dir) / f"{table_name}_topic_similarity.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(similarity_results, f, indent=2, ensure_ascii=False)

                _console.print_status(f"Topic similarity results saved to: {output_file}", "success")

                # Generate visualizations
                _console.print_status("Generating topic similarity visualizations...", "processing")

                viz_paths = plot_combined_similarity_analysis(
                    similarity_matrix=similarity_matrix,
                    topic_names=topic_names,
                    output_dir=str(table_output_dir),
                    dataset_name=table_name,
                    threshold=0.5,
                    create_network=True,
                    create_dendrogram=True,
                    create_distribution=True
                )

                _console.print_status("Topic similarity analysis completed!", "success")

            except Exception as e:
                _console.print_warning(f"Could not compute topic similarity: {str(e)}", tag="SIMILARITY")

        # Calculate reconstruction error

        # X_reconstructed = nmf_output["W"] @ nmf_output["H"]
        # frobenius_norm = np.linalg.norm(tdm - X_reconstructed, 'fro')

        #A,L  = build_correlation_graph(nmf_output["W"])


        return topic_word_scores, topic_doc_scores, coherence_scores, nmf_output, word_result
