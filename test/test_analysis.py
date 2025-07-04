import nmf_standalone

print(dir(nmf_standalone))
emj_map = nmf_standalone.EmojiMap()


file_path = "../veri_setleri/APPSTORE_APP_REVIEWSyeni_yeni.csv"
column = "REVIEW"
nmf_standalone.run_topic_analysis(filepath=file_path,
                                  column=column,
                                  language="TR",
                                  tokenizer_type="bpe",
                                  lemmatize=True,
                                  generate_wordclouds=True,
                                  topic_count=8,
                                  words_per_topic=15,
                                  emoji_map=emj_map,
                                  word_pairs_out=False,
                                  separator='|')

