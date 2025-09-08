import manta

if __name__ == '__main__':
    file_path = "../veri_setleri/covid_abstracts.csv"
    column = "abstract"

    result = manta.run_topic_analysis(
        filepath=file_path,
        column=column,
        separator=",",
        language="EN",
        lemmatize=True,
        topic_count=10,
        words_per_topic=15,
        nmf_method="nmf", # "nmf" or "nmtf" or "pnmf"
        filter_app=False,
        data_filter_options = {
            "filter_app_country": "TR",
            "filter_app_country_column": "REVIEWER_LANGUAGE",
        },
        emoji_map=True,
        generate_wordclouds=True,
        save_to_db=False,
        word_pairs_out=False,
        topic_distribution=True,
    )