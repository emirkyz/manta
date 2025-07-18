import manta

print(dir(manta))


file_path = "../veri_setleri/bbc_news.csv"
column = "text"
manta.run_topic_analysis(filepath=file_path,
                                  column=column,
                                  separator=',',
                                  language="EN",
                                  tokenizer_type="bpe",
                                  lemmatize=True,
                                  generate_wordclouds=True,
                                  topic_count=10,
                                  words_per_topic=15,
                                  emoji_map=True,
                                  word_pairs_out=False,
                                  nmf_method="nmf")

