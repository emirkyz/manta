

from ..analysis.word_cooccurrence import calc_word_cooccurrence
from ..analysis.word_cooccurrence_analyzer import analyze_word_cooccurrence


def create_visualization(W, H, sozluk, table_output_dir, table_name, options, result, topic_word_scores, metin_array, topics_db_eng, emoji_map, program_output_dir, output_dir):
    # generate topic distribution plot
    topic_dist_img_count = 0
    if options["gen_topic_distribution"]:
        from .topic_dist import gen_topic_dist
        topic_dist_img_count = gen_topic_dist(W, table_output_dir, table_name)

    
    # generate t-SNE visualization plot
    if False:
        from .tsne_graph_output import tsne_graph_output
        tsne_plot_path = tsne_graph_output(
            w=W,
            h=H,
            output_dir=table_output_dir,
            table_name=table_name
        )

    # generate topic-space fuzzy classification plot
    from .topic_space_graph_output_old import topic_space_graph_output
    topic_space_plot_path = topic_space_graph_output(
        w=W,
        h=H,
        output_dir=table_output_dir,
        table_name=table_name,
        top_k=3,
        min_probability=0,
        positioning="radial"
    )

    # generate interactive LDAvis-style visualization
    if False:
        from .manta_ldavis_output import create_manta_ldavis
        ldavis_plot_path = create_manta_ldavis(
            w_matrix=W,
            h_matrix=H,
            vocab=sozluk,
            output_dir=table_output_dir,
            table_name=table_name
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
            print(f"Using sliding window co-occurrence analysis with options")
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
                H, sozluk, table_output_dir, table_name, 
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