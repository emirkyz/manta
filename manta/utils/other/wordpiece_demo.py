from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF # Using sklearn's NMF


# For WordPiece (requires 'tokenizers' library)
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

# --- 1. Example Data ---
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Never jump over the lazy dog again.",
    "The fox is quick.",
    "A quick brown dog is good.",
    "Apples are healthy fruit.",
    "Oranges are citrus fruit.",
    "Bananas are yellow fruit.",
    "The weather is very hot and sunny today.",
    "The climate change is a global concern.",
    "Green apples are sour but red apples are sweet."
]

# --- Main execution block ---
if __name__ == '__main__':
    # --- 2. Preprocessing for NMF (Word-level tokenization) ---
    processed_texts_for_nmf = []
    for doc in documents:
        tokens = [word for word in doc.lower().replace('.', '').split() if word.isalpha()]
        processed_texts_for_nmf.append(tokens)

    # Create dictionary and corpus for Gensim's CoherenceModel
    dictionary = Dictionary(processed_texts_for_nmf)
    corpus = [dictionary.doc2bow(text) for text in processed_texts_for_nmf]

    # --- 3. Train NMF Model (using sklearn's NMF) ---
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    num_topics = 3 # Choose your desired number of topics
    sklearn_nmf_model = NMF(n_components=num_topics, random_state=42, init='nndsvda', max_iter=200)
    sklearn_nmf_model.fit(tfidf_matrix)

    # --- 4. Extract Top Words per Topic for Coherence Calculation ---
    feature_names = tfidf_vectorizer.get_feature_names_out()
    nmf_topics = []
    num_top_words = 10 # Number of top words to consider for coherence
    for topic_idx, topic in enumerate(sklearn_nmf_model.components_):
        top_words_idx = topic.argsort()[:-num_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        nmf_topics.append(top_words)
        print(f"Topic {topic_idx + 1} (NMF): {', '.join(top_words)}")

    # --- 5. Calculate Coherence Score using Gensim ---
    coherence_model_nmf = CoherenceModel(
        topics=nmf_topics,
        texts=processed_texts_for_nmf,
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence_nmf = coherence_model_nmf.get_coherence()
    print(f"\nC_v Coherence Score for NMF: {coherence_nmf:.4f}")

    coherence_per_topic = coherence_model_nmf.get_coherence_per_topic()
    print(f"C_v Coherence per topic for NMF: {coherence_per_topic}")

    print("-" * 50)

    # --- Corrected WordPiece Tokenizer Setup and Training ---

    wordpiece_tokenizer = Tokenizer(WordPiece(unk_token="[BİLİNMİYOR]"))
    wordpiece_tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=30000,
        min_frequency=2,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[BİLİNMİYOR]"]
        # show_progress is NOT a parameter of the trainer either.
        # Progress is usually handled by the library's internal logging or is not directly exposed.
    )

    training_corpus_iterator = (doc for doc in documents)
    wordpiece_tokenizer.train_from_iterator(
        training_corpus_iterator,
        trainer=trainer,
        # REMOVED: show_progress=True
    )

    wordpiece_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", wordpiece_tokenizer.token_to_id("[CLS]")),
            ("[SEP]", wordpiece_tokenizer.token_to_id("[SEP]")),
        ],
    )

    print("\n--- Demonstrating WordPiece Tokenization (NOT for Coherence) ---")
    example_sentence = "The quick brown fox jumps over the lazy dog."
    encoded_output = wordpiece_tokenizer.encode(example_sentence)

    print(f"Original: {example_sentence}")
    print(f"WordPiece Tokens: {encoded_output.tokens}")
    print(f"WordPiece IDs: {encoded_output.ids}")

    example_sentence_turkish_oov = "Merhaba Dünya, bu bir test cümlesidir ve bilinmiyor kelimesi."
    encoded_output_turkish = wordpiece_tokenizer.encode(example_sentence_turkish_oov)
    print(f"Original (Turkish with OOV): {example_sentence_turkish_oov}")
    print(f"WordPiece Tokens: {encoded_output_turkish.tokens}")
    print(f"WordPiece IDs: {encoded_output_turkish.ids}")