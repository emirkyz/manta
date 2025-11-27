"""
Topic modeling pipeline for MANTA topic analysis.
"""

from typing import Dict, Any, Optional, Tuple

import pandas as pd

from .._functions.common_language.topic_extractor import topic_extract
from .._functions.nmf import run_nmf
from ..utils.analysis.coherence_score import calculate_coherence_scores
from ..utils.export.save_doc_score_pair import save_doc_score_pair
from ..utils.export.save_word_score_pair import save_word_score_pair
from ..utils.export.save_s_matrix import save_s_matrix
from ..utils.console.console_manager import ConsoleManager


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
        if console:
            console.print_status(f"Starting NMF processing ({options['nmf_type'].upper()})...", "processing")
        else:
            print("Starting NMF processing...")
        
        # NMF processing
        nmf_output = run_nmf(
            num_of_topics=int(options["DESIRED_TOPIC_COUNT"]),
            sparse_matrix=tdm,
            norm_thresh=0.005,
            nmf_method=options["nmf_type"],
        )

        if console:
            console.print_status("Extracting topics from NMF results...", "processing")
        else:
            print("Generating topic groups...")
            
        # Extract topics based on language
        if options["LANGUAGE"] == "TR":
            word_result, document_result = topic_extract(
                H=nmf_output["H"],
                W=nmf_output["W"],
                doc_word_pairs=nmf_output.get("S", None),
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
                doc_word_pairs=nmf_output.get("S", None),
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

        if console:
            console.print_status("Saving topic results...", "processing")
        else:
            print("Saving topic results...")
            
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
            if console:
                console.print_status("Saving S matrix...", "processing")
            save_s_matrix(
                s_matrix=nmf_output["S"],
                output_dir=table_output_dir,
                table_name=table_name,
                data_frame_name=table_name,
            )

            # Generate S matrix graph visualizations
            if console:
                console.print_status("Generating S matrix graph visualizations...", "processing")
            else:
                print("Generating S matrix graph visualizations...")

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

        if console:
            console.print_status("Calculating coherence scores...", "processing")
        else:
            print("Calculating coherence scores...")
            
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

        return topic_word_scores, topic_doc_scores, coherence_scores, nmf_output, word_result
