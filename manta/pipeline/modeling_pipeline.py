"""Topic modeling pipeline for MANTA topic analysis."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .._functions.common_language.topic_extractor import topic_extract
from .._functions.nmf import run_nmf
from ..config import TopicAnalysisConfig
from ..utils.analysis.gensim_coherence import calculate_gensim_cv_coherence
from ..utils.export.save_doc_score_pair import save_doc_score_pair
from ..utils.export.save_word_score_pair import save_word_score_pair
from ..utils.export.save_s_matrix import save_s_matrix
from ..utils.console.console_manager import ConsoleManager, get_console
from ..utils.processing_utils import CachedData, PipelineContext


class ModelingPipeline:
    """Handles NMF topic modeling and analysis."""

    @staticmethod
    def perform_topic_modeling(
        cached_data: CachedData,
        config: TopicAnalysisConfig,
        ctx: PipelineContext,
        table_output_dir: Path,
    ) -> Tuple[Dict, Dict, Dict, Dict, Any]:
        """Perform NMF topic modeling and analysis.

        Args:
            cached_data: TF-IDF matrix, vocabulary, and text arrays from the data stage
            config: Typed analysis configuration
            ctx: Pipeline context with paths, db config, console, and runtime state
            table_output_dir: Directory for writing output files

        Returns:
            Tuple of (topic_word_scores, topic_doc_scores, coherence_scores, nmf_output, word_result)
        """
        _console = ctx.console or get_console()
        table_name = ctx.paths.table_name
        _console.print_status(f"Starting NMF processing ({config.nmf_method.upper()})...", "processing")

        if config.barebones:
            from ..utils.analysis.gensim_coherence import extract_relevance_top_words
            nmf_output = run_nmf(
                num_of_topics=int(config.topic_count),
                sparse_matrix=cached_data.tdm,
                norm_thresh=0.005,
                nmf_method=config.nmf_method,
            )
            _console.print_status("Extracting relevance-scored top words (barebones mode)...", "processing")
            relevance_top_words = extract_relevance_top_words(
                h_matrix=nmf_output["H"],
                w_matrix=nmf_output["W"],
                vocabulary=cached_data.vocab,
                s_matrix=nmf_output.get("S"),
                lambda_val=0.6,
                top_n=config.words_per_topic,
            )
            coherence_scores = {"relevance": relevance_top_words}
            if table_output_dir and table_name:
                output_path = Path(table_output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                coherence_file = output_path / f"{table_name}_relevance_top_words.json"
                with open(coherence_file, "w", encoding="utf-8") as f:
                    json.dump(coherence_scores, f, indent=4, ensure_ascii=False)
                _console.print_debug(f"Relevance top words saved to: {coherence_file}", tag="COHERENCE")
            return {}, {}, coherence_scores, nmf_output, None

        nmf_output = run_nmf(
            num_of_topics=int(config.topic_count),
            sparse_matrix=cached_data.tdm,
            norm_thresh=0.005,
            nmf_method=config.nmf_method,
        )

        _console.print_status("Extracting topics from NMF results...", "processing")

        extract_kwargs = dict(
            H=nmf_output["H"],
            W=nmf_output["W"],
            s_matrix=nmf_output.get("S"),
            topic_count=int(config.topic_count),
            vocab=cached_data.vocab,
            documents=cached_data.text_array,
            original_documents=cached_data.original_text_array,
            db_config=ctx.db_config,
            data_frame_name=table_name,
            word_per_topic=config.words_per_topic,
            include_documents=True,
            emoji_map=ctx.emoji_map,
        )
        if config.language == "TR":
            extract_kwargs["tokenizer"] = ctx.tokenizer

        word_result, document_result = topic_extract(**extract_kwargs)

        _console.print_status("Saving topic results...", "processing")

        topic_word_scores = save_word_score_pair(
            base_dir=None,
            output_dir=table_output_dir,
            table_name=table_name,
            topics_data=word_result,
            result=None,
            data_frame_name=table_name,
            topics_db_eng=ctx.db_config.topics_db_engine,
        )

        topic_doc_scores = save_doc_score_pair(
            document_result,
            base_dir=None,
            output_dir=table_output_dir,
            table_name=table_name,
            data_frame_name=table_name,
        )

        if "S" in nmf_output:
            _console.print_status("Saving S matrix...", "processing")
            save_s_matrix(
                s_matrix=nmf_output["S"],
                output_dir=table_output_dir,
                table_name=table_name,
                data_frame_name=table_name,
            )
            _console.print_status("Generating S matrix graph visualizations...", "processing")
            from ..utils.visualization.s_matrix_graph import visualize_s_matrix_graph
            visualize_s_matrix_graph(
                s_matrix=nmf_output["S"],
                output_dir=table_output_dir,
                table_name=table_name,
                threshold=0.01,
                layout="circular",
                create_interactive=False,
                create_heatmap=True,
            )

        _console.print_status("Calculating coherence scores...", "processing")

        coherence_results = calculate_gensim_cv_coherence(
            h_matrix=nmf_output["H"],
            w_matrix=nmf_output["W"],
            vocabulary=cached_data.vocab,
            documents=cached_data.text_array,
            s_matrix=nmf_output.get("S"),
            lambda_val=0.6,
            top_n_words=config.words_per_topic,
        )

        coherence_scores = {
            "relevance": coherence_results["topic_word_scores"],
            "gensim": {
                "c_v_average": coherence_results["c_v_average"],
                "c_v_per_topic": coherence_results["c_v_per_topic"],
                "u_mass_average": coherence_results["u_mass_average"],
                "u_mass_per_topic": coherence_results["u_mass_per_topic"],
            },
        }

        _console.print_status("Calculating simplified silhouette score...", "processing")
        try:
            import gc
            from ..utils.analysis.silhouette import calculate_simplified_silhouette
            silhouette_result = calculate_simplified_silhouette(nmf_output["W"])
            gc.collect()
            coherence_scores["silhouette"] = silhouette_result
            if silhouette_result["average"] is not None:
                _console.print_status(
                    f"Silhouette score: {silhouette_result['average']:.4f} "
                    f"({silhouette_result['n_assigned']}/{silhouette_result['n_total']} docs assigned)",
                    "success",
                )
        except Exception as e:
            _console.print_status(f"Silhouette calculation skipped: {e}", "warning")

        # Save coherence results to JSON (includes relevance top words and silhouette)
        if table_output_dir and table_name:
            output_path = Path(table_output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            coherence_file = output_path / f"{table_name}_relevance_top_words.json"
            with open(coherence_file, "w", encoding="utf-8") as f:
                json.dump(coherence_scores, f, indent=4, ensure_ascii=False)
            _console.print_debug(f"Coherence scores saved to: {coherence_file}", tag="COHERENCE")

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
