from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix

import manta
from manta._functions.english.english_preprocessor import clean_english_text
from manta._functions.english.english_text_encoder import counterize_english
from manta._functions.tfidf import tf_l


def load_model_components(model_file="topic_model_components.npz"):
    """Load saved model components from npz file."""
    try:
        data = np.load(model_file, allow_pickle=True)
        W = data['W']
        H = data['H']
        vocab = data['vocab']

        # Load S matrix if present (NMTF models)
        S = data['S'] if 'S' in data else None

        # Handle different vocab formats - convert to list if needed
        if isinstance(vocab, np.ndarray):
            if vocab.ndim == 0:  # scalar array (single item)
                vocab = vocab.item()
            else:  # array of items
                vocab = vocab.tolist()

        print(f"Model components loaded from {model_file}")
        print(f"Vocabulary type: {type(vocab)}, length: {len(vocab) if hasattr(vocab, '__len__') else 'N/A'}")
        return W, H, vocab, S
    except FileNotFoundError:
        print(f"Model file {model_file} not found. Run training first.")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None


def predict_topics(text, model_file="topic_model_components.npz", top_n=3, normalization='l1'):
    """
    Predict topics for new text using saved model components.

    Args:
        text (str): Input text to analyze
        model_file (str): Path to saved model components
        top_n (int): Number of top topics to return
        normalization (str): Normalization method - 'l1', 'l2', 'minmax', 'softmax', or 'none'

    Returns:
        dict: Dictionary with raw scores and normalized scores using specified method
    """
    # Load model components
    W, H, vocab, S = load_model_components(model_file)
    if H is None or vocab is None:
        return None

    # Preprocess the input text
    sentence = clean_english_text([text], lemmatize=True)
    data = counterize_english(vocab, sentence, lemmatize=True)

    if not data or not data[0]:  # Check if preprocessing resulted in empty data
        print("No valid tokens found in input text")
        return None

    # Create document-term matrix
    document_count = len(data)
    vocabulary_count = len(vocab)
    matris = lil_matrix((document_count, vocabulary_count), dtype=int)

    for i, document in enumerate(data):
        histogram = Counter(document)
        temporary = [(k, v) for k, v in histogram.items()]
        columns = [a[0] for a in temporary]
        values = [b[1] for b in temporary]
        matris[i, columns] = values

    # Calculate TF-IDF
    input_matrix = matris.tocsc(copy=True)
    input_matrix.data = np.ones_like(input_matrix.data)

    idf = 1
    input_matrix = input_matrix.tocsr()
    tf = tf_l(input_matrix)
    tf_idf = tf.multiply(idf)

    # Normalize
    tf_idf = tf_idf.tocsr()
    from sklearn.preprocessing import normalize
    tf_idf = normalize(tf_idf, norm='l2', axis=1)

    # Predict topic distribution
    raw_scores = H @ tf_idf.T

    # Apply L1 normalization (better alternative to softmax for topic modeling)
    raw_scores_flat = raw_scores.flatten()  # Convert to 1D array for easier processing
    l1_scores = np.abs(raw_scores_flat) / np.sum(np.abs(raw_scores_flat))

    # Get top topics for raw scores
    raw_labeled_results = [(f"Topic {i+1}", raw_scores_flat[i]) for i in range(len(raw_scores_flat))]
    raw_sorted_results = sorted(raw_labeled_results, key=lambda x: x[1], reverse=True)

    # Get top topics for L1 normalized scores
    l1_labeled_results = [(f"Topic {i+1}", l1_scores[i]) for i in range(len(l1_scores))]
    l1_sorted_results = sorted(l1_labeled_results, key=lambda x: x[1], reverse=True)

    return {
        'raw_scores': raw_sorted_results[:top_n],
        'l1_scores': l1_sorted_results[:top_n]
    }


if __name__ == '__main__':

    # Get data from PostgreSQL database and convert to CSV
    import psycopg2
    import pandas as pd
    import os

    # Database connection parameters
    db_config = {
        'host': 'localhost',
        'database': 'turkcell',
        'user': 'emir_dxhub',
        'password': '123',
        'port': 5432
    }

    # SQL query to fetch data
    sql_query = """select distinct rev_submit_millis_since_epoch, device,review_text
            from dxhub.playstore_app_reviews
            where package_name = 'com.turkcell.ott'
            and reviewer_language = 'tr'
            order by rev_submit_millis_since_epoch desc"""



    try:
        # Connect to PostgreSQL database
        print("Connecting to PostgreSQL database...")
        conn = psycopg2.connect(**db_config)

        # Execute query and load into DataFrame
        print("Executing SQL query...")
        df = pd.read_sql_query(sql_query, conn)

        # Close database connection
        conn.close()
        print(f"Successfully fetched {len(df)} records from database")

        # Create output directory if it doesn't exist
        output_dir = "../veri_setleri"
        os.makedirs(output_dir, exist_ok=True)

        # Save to CSV
        csv_file_path = os.path.join(output_dir, "database_export.csv")
        df.to_csv(csv_file_path, index=False, sep="|", encoding='utf-8')
        print(f"Data exported to: {csv_file_path}")

        # Update file_path to use the exported CSV
        file_path = csv_file_path

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        print("Falling back to original CSV file...")
        file_path = "../veri_setleri/playstore.csv"
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to original CSV file...")
        file_path = "../veri_setleri/playstore.csv"



    def get_df_from_db(db_path:Path, table_name:str):
        """Load df from SQLite database."""

        connection_string = f"sqlite:///{db_path}"
        engine = manta.create_engine(connection_string)
        df = pd.read_sql_table(table_name, engine)
        # convert to csv

        return df



    file_path = "../misc/DNA/combined_dna_sequence.csv"
    file_path = "../veri_setleri/abstracts_pubmed_v2.csv"
    file_path = "../veri_setleri/radiology_imaging.csv"


    column = "abstract"
    result = manta.run_topic_analysis(
        filepath=file_path,
        column=column,
        separator=",",
        language="EN",
        lemmatize=True,
        topic_count=15,
        words_per_topic=15,
        nmf_method="nmtf", # "nmf" or "nmtf" or "pnmf"
        tokenizer_type="bpe",
        filter_app=False,
        data_filter_options = {
            "filter_app_country": "TR",
            "filter_app_country_column": "reviewer_language",
            "filter_app_name":"com.turkcell.ott",
            "filter_app_column": "package_name",
        },
        emoji_map=False,
        generate_wordclouds=True,
        save_to_db=False,
        word_pairs_out=False,
        topic_distribution=True,
        export_excel=False,
        output_dir="weighted_time_test",
    )

    if False:
        old_str = """
        Manchester City secured a crucial 2-1 victory over Arsenal at the Emirates Stadium, extending their lead at the top of the Premier League table to five points. Erling Haaland opened the scoring in the 23rd minute with a trademark finish, capitalizing on a perfectly weighted through ball from Kevin De Bruyne.
         claim their third consecutive Premier League title. Ayrıca bu kazanç, ingiltere liginde ilgiye sebep oldu
         """
        str ="""
        COVID-19 pandemic has underscored the need for a well-trained public health workforce to save lives through timely outbreaks detection and response  In Yemen  a country that is entering its seventh year of a protracted war  the ongoing conflict severely limited the country s capacity to implement effective preparedness and response measures to outbreaks including COVID-19  There are growing concerns that the virus may be circulating within communities undetected and unmitigated especially as underreporting continues in some areas of the country due to a lack of testing facilities  delays in seeking treatment  stigma  difficulty accessing treatment centers  the perceived risks of seeking care or for political issues  The Yemen Field Epidemiology Training Program  FETP  was launched in 2011 to address the shortage of a skilled public health workforce  with the objective of strengthening capacity in field epidemiology  Thus  events of public health importance can be detected and investigated in a timely and effective manner  During the COVID-19 pandemic  the Yemen FETP s response has been instrumental through participating in country-level coordination  planning  monitoring  and developing guidelines standard operating procedures and strengthening surveillance capacities  outbreak investigations  contact tracing  case management  infection prevention  and control  risk communication  and research  As the third wave is circulating with a steeper upward curve than the previous ones with possible new variants  the country will not be able to deal with a surge of cases as secondary care is extremely crippled  Since COVID-19 prevention and control are the only option available to reduce its grave impact on morbidity and mortality  health partners should support the Yemen FETP to strengthen the health system s response to future epidemics  One important lesson learned from the COVID-19 pandemic  especially in the Yemen context and applicable to developing and war-torn countries  is that access to outside experts becomes limited  therefore  it is crucial to invest in building national expertise to provide timely  cost-effective  and sustainable services that are culturally appropriate  It is also essential to build such expertise at the governorate and district levels  as they are normally the first respondents  and to provide them with the necessary tools for immediate response in order to overcome the disastrous delays
        """

        model_file = result.get("model_file", None)

        print("\n=== Testing save/load workflow ===")
        # Test the new prediction function with the same text
        test_predictions = predict_topics(str, model_file=model_file, top_n=5)
        if test_predictions:
            print("Top predictions using saved model:")
            for i in range(len(test_predictions['raw_scores'])):
                raw_topic, raw_score = test_predictions['raw_scores'][i]
                l1_topic, l1_score = test_predictions['l1_scores'][i]
                print(f"{raw_topic} raw: {raw_score:.4f} L1: {l1_score:.4f}")
