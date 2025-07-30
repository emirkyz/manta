import manta

file_path = "../veri_setleri/findings.csv"
column = "findings"

result = manta.run_topic_analysis(
    filepath=file_path,
    column=column,
    separator=',',
    language="EN",
    tokenizer_type="bpe",
    lemmatize=True,
    generate_wordclouds=True,
    topic_count=10,
    words_per_topic=15,
    emoji_map=False,
    word_pairs_out=False,
    nmf_method="nmtf",
    filter_app=False,
    data_filter_options = {
        "filter_app_country": "TR",
        "filter_app_country_column": "REVIEWER_LANGUAGE",
    },
)