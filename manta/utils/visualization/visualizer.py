from . import visualize_s_matrix_graph
from ..analysis.word_cooccurrence import calc_word_cooccurrence
from ..analysis.word_cooccurrence_analyzer import analyze_word_cooccurrence
from ..export.save_s_matrix import _normalize_s_matrix_columns
from ..console.console_manager import ConsoleManager, get_console
import logging

logger = logging.getLogger(__name__)


def create_visualization(nmf_output, sozluk, table_output_dir, table_name, options, result, topic_word_scores, metin_array, topics_db_eng, emoji_map, program_output_dir, output_dir, datetime_series=None, console=None):
    _console = console or get_console()
    # Normalize S matrix if present (for NMTF)
    if "S" in nmf_output and nmf_output["S"] is not None:
        logger.info("Normalizing S matrix for visualizations (L1 column normalization)")
        # Store both versions in nmf_output
        nmf_output["S_original"] = nmf_output["S"]
        nmf_output["S"] = _normalize_s_matrix_columns(nmf_output["S"])
        logger.info("S matrix normalized - visualizations will use normalized version")

    # generate topic distribution plot
    topic_dist_img_count = 0
    if options["gen_topic_distribution"]:
        from .topic_dist import gen_topic_dist
        topic_dist_img_count = gen_topic_dist(nmf_output["W"], table_output_dir, table_name, s_matrix=nmf_output.get("S", None))

    
    # generate t-SNE visualization plot
    if options["gen_tsne"]:
        # Use optimized t-SNE for large datasets (>5K documents)
        n_docs = nmf_output["W"].shape[0]
        use_optimized = False
        
        if use_optimized:
            try:
                from .tsne_optimized import tsne_graph_output_optimized
                _console.print_debug(f"Using optimized t-SNE for {n_docs:,} documents", tag="VISUALIZATION")
                tsne_plot_path = tsne_graph_output_optimized(
                    w=nmf_output["W"],
                    h=nmf_output["H"],
                    s_matrix=nmf_output.get("S", None),
                    output_dir=table_output_dir,
                    table_name=table_name,
                    performance_mode="auto"
                )
            except ImportError as e:
                _console.print_warning(f"Optimized t-SNE not available, falling back to standard: {e}", tag="VISUALIZATION")
                from .tsne_graph_output import tsne_graph_output
                tsne_plot_path = tsne_graph_output(
                    w=nmf_output["W"],
                    h=nmf_output["H"],
                    s_matrix=nmf_output.get("S", None),
                    output_dir=table_output_dir,
                    table_name=table_name
                )
        else:
            from .tsne_graph_output import tsne_graph_output
            tsne_plot_path = tsne_graph_output(
                w=nmf_output["W"],
                h=nmf_output["H"],
                s_matrix=nmf_output.get("S", None),
                output_dir=table_output_dir,
                table_name=table_name,
            )

    # generate topic-space fuzzy classification plot
    from .topic_space_graph_output_old import topic_space_graph_output

    if False:
        topic_space_plot_path = topic_space_graph_output(
        w=nmf_output["W"],
        h=nmf_output["H"],
        s_matrix=nmf_output.get("S", None),
        output_dir=table_output_dir,
        table_name=table_name,
        top_k=3,
        min_probability=0,
        positioning="radial"
    )

    if True and "S" in nmf_output:
        paths = visualize_s_matrix_graph(
            s_matrix=nmf_output["S"],
            output_dir=table_output_dir,
            table_name="my_analysis",
            threshold=0.1,  # Filter edges below this value
            layout="circular"  # or "spring", "kamada_kawai"
        )

    # generate temporal topic distribution plot
    if datetime_series is not None and len(datetime_series) > 0:
        from .topic_temporal_dist import gen_temporal_topic_dist
        import pandas as pd

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(datetime_series):
            # Detect format based on column name
            datetime_col = options.get('datetime_column', '')
            if datetime_col is None:
                logger.warning("No datetime column specified in options. Using empty string.")
                datetime_col_name = ''
            else:
                datetime_col_name = datetime_col.lower()

            if 'millis' in datetime_col_name or 'epoch' in datetime_col_name:
                datetime_series = pd.to_datetime(datetime_series, unit='ms')
            elif 'year' in datetime_col_name and not options.get('datetime_is_combined_year_month', False):
                datetime_series = pd.to_datetime(datetime_series, format='%Y')
            else:
                datetime_series = pd.to_datetime(datetime_series)

        # Determine appropriate time grouping based on datetime type
        # Use 'month' grouping for combined year/month columns, otherwise use 'year'
        time_grouping = 'month' if options.get('datetime_is_combined_year_month', False) else 'year'

        try:
            time_grouping = "quarter"

            fig, temporal_df = gen_temporal_topic_dist(
                W=nmf_output["W"],
                s_matrix=nmf_output.get("S", None),
                datetime_series=datetime_series,
                use_weighted=True,
                output_dir=table_output_dir,
                table_name=table_name,
                time_grouping='quarter',  # Options: 'year', 'month', 'quarter', 'week'
                plot_type='stacked_area',  # Options: 'stacked_area', 'line', 'heatmap', 'stacked_bar'
                normalize=False,  # False for count-based, True for percentage-based
                min_score=0.0
            )

            fig, temporal_df = gen_temporal_topic_dist(
                W=nmf_output["W"],
                s_matrix=nmf_output.get("S", None),
                datetime_series=datetime_series,
                output_dir=table_output_dir,
                table_name=table_name,
                use_weighted=True,
                time_grouping="quarter",  # Options: 'year', 'month', 'quarter', 'week'
                plot_type='line',  # Options: 'stacked_area', 'line', 'heatmap', 'stacked_bar'
                normalize=False,  # False for count-based, True for percentage-based
                min_score=0.0,
                use_mm_yyyy_format=options.get('datetime_is_combined_year_month', False)
            )
            
            if options.get("gen_line_html",False):
                from .create_interactive_temporal import generate_temporal_line_graph
                generate_temporal_line_graph(
                    CSV_PATH=table_output_dir / f"{table_name}_temporal_topic_dist_{time_grouping}.csv",
                    OUTPUT_HTML=table_output_dir / f"{table_name}_temporal_topic_distribution.html"
                )
            
            
            _console.print_debug(f"Generated temporal topic distribution visualization", tag="VISUALIZATION")
        except Exception as e:
            _console.print_warning(f"Failed to generate temporal visualization: {e}", tag="VISUALIZATION")

        # Generate violin plot showing topic distribution by year
        try:
            from .violin_plot import gen_violin_plot
            violin_path = gen_violin_plot(
                W=nmf_output["W"],
                S_matrix=nmf_output.get("S", None),
                datetime_series=datetime_series,
                table_output_dir=table_output_dir,
                table_name=table_name
            )
            _console.print_debug(f"Generated topic distribution violin plot by year: {violin_path.name}", tag="VISUALIZATION")

        except Exception as e:
            _console.print_warning(f"Failed to generate violin plot: {e}", tag="VISUALIZATION")

    # generate interactive LDAvis-style visualization
    if False:
        from .manta_ldavis_output import create_manta_ldavis
        ldavis_plot_path = create_manta_ldavis(
            w_matrix=nmf_output["W"],
            h_matrix=nmf_output["H"],
            s_matrix=nmf_output.get("S", None),
            vocab=sozluk if options["LANGUAGE"] == "EN" else None,
            output_dir=table_output_dir,
            table_name=table_name,
            tokenizer=options["tokenizer"] if options["LANGUAGE"] == "TR" else None,
        )

    

    if options["gen_cloud"]:
        from .gen_cloud import generate_wordclouds
        generate_wordclouds(result, table_output_dir, table_name)

    if options["save_excel"]:
        from ..export.export_excel import export_topics_to_excel
        export_topics_to_excel(topic_word_scores, table_output_dir, table_name)

    if options["word_pairs_out"]:
        # Choose between old NMF-based co-occurrence and new sliding window co-occurrence
        cooccurrence_method = "sliding_window"   # Default to old method for backward compatibility
        
        if cooccurrence_method == "sliding_window":
            _console.print_debug(f"Using sliding window co-occurrence analysis with options", tag="VISUALIZATION")
            # Use new memory-efficient sliding window co-occurrence analyzer
            language = "turkish" if options["LANGUAGE"] == "TR" else "english"
            top_pairs = analyze_word_cooccurrence(
                input_data=metin_array,
                window_size=options.get("cooccurrence_window_size", 5),
                min_count=options.get("cooccurrence_min_count", 2),
                max_vocab_size=options.get("cooccurrence_max_vocab", None),
                output_dir=str(table_output_dir),  # Use the table output dir directly
                table_name=table_name,
                language=language,
                create_heatmap=True,
                heatmap_size=options.get("cooccurrence_heatmap_size", 20),
                top_n=options.get("cooccurrence_top_n", 100),
                batch_size=options.get("cooccurrence_batch_size", 1000),
                create_output_folder=False  # Don't create extra Output folder
            )
        else:
            # Use original NMF-based co-occurrence (default behavior)
            top_pairs = calc_word_cooccurrence(
                nmf_output["H"], sozluk, table_output_dir, table_name,
                top_n=options.get("cooccurrence_top_n", 100), 
                min_score=options.get("cooccurrence_min_score", 1),
                language=options["LANGUAGE"], 
                tokenizer=options["tokenizer"],
                create_heatmap=True
            )

    '''new_hierarchy = hierarchy_nmf(W, tdm, selected_topic=1, desired_topic_count=options["DESIRED_TOPIC_COUNT"],
                                    nmf_method=options["nmf_type"], sozluk=sozluk, tokenizer=tokenizer,
                                    metin_array=metin_array, topics_db_eng=topics_db_eng, table_name=table_name,
                                    emoji_map=emoji_map, base_dir=program_output_dir, output_dir=output_dir)'''

    return topic_dist_img_count if options["gen_topic_distribution"] else None