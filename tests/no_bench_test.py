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
    topic_count=5,
    words_per_topic=15,
    emoji_map=False,
    word_pairs_out=False,
    nmf_method="nmf",
    filter_app=False,
)