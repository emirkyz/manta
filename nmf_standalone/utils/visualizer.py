from .gen_cloud import generate_wordclouds
from .export_excel import export_topics_to_excel
from .coherence_score import calculate_coherence_scores
from .save_word_score_pair import save_word_score_pair
from .topic_dist import gen_topic_dist
from .word_cooccurrence import calc_word_cooccurrence


def create_visualization(W, H, sozluk, table_output_dir, table_name, options, result, topic_word_scores, metin_array, topics_db_eng, emoji_map, program_output_dir, output_dir):
    # generate topic distribution plot
    if options["gen_topic_distribution"]:
        gen_topic_dist(W, table_output_dir, table_name)

    if options["gen_cloud"]:
        generate_wordclouds(result, table_output_dir, table_name)

    if options["save_excel"]:
        export_topics_to_excel(topic_word_scores, table_output_dir, table_name)

    if options["word_pairs_out"]:
        # Calculate word co-occurrence matrix and save to output dir
        top_pairs = calc_word_cooccurrence(H, sozluk, table_output_dir, table_name, top_n=100, min_score=1,
                                            language=options["LANGUAGE"], tokenizer=options["tokenizer"],create_heatmap=True)

    '''new_hierarchy = hierarchy_nmf(W, tdm, selected_topic=1, desired_topic_count=options["DESIRED_TOPIC_COUNT"],
                                    nmf_method=options["nmf_type"], sozluk=sozluk, tokenizer=tokenizer,
                                    metin_array=metin_array, topics_db_eng=topics_db_eng, table_name=table_name,
                                    emoji_map=emoji_map, base_dir=program_output_dir, output_dir=output_dir)'''
