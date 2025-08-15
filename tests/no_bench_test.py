import manta

file_path = "../veri_setleri/PLAYSTORE_REVIEWS_yeni.csv"
column = "REVIEW_TEXT"

result = manta.run_topic_analysis(
    filepath=file_path,
    column=column,
    separator='|',
    language="TR",
    tokenizer_type="bpe",
    lemmatize=True,
    generate_wordclouds=True,
    topic_count=10,
    words_per_topic=15,
    emoji_map=True,
    word_pairs_out=False,
    nmf_method="nmtf", # "nmf" or "nmtf" or "pnmf"
    filter_app=True,
    data_filter_options = {
        "filter_app_country": "TR",
        "filter_app_country_column": "REVIEWER_LANGUAGE",
    },
    save_to_db=False
)