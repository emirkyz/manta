import nmf_standalone

print(dir(nmf_standalone))
emj_map = nmf_standalone.EmojiMap()


file_path = "../veri_setleri/PLAYSTORE_APP_REVIEWS.csv"
column = "REVIEW_TEXT"
nmf_standalone.run_topic_analysis(filepath=file_path,
                                  column=column,
                                  separator=';',
                                  language="TR",
                                  tokenizer_type="bpe",
                                  lemmatize=True,
                                  generate_wordclouds=True,
                                  topic_count=5,
                                  words_per_topic=15,
                                  emoji_map=emj_map,
                                  word_pairs_out=True
                                  ,nmf_method="opnmf")

