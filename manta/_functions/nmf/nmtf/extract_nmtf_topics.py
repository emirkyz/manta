import numpy as np
from scipy import sparse as sp


def extract_topics(doc_by_topics: sp.csr_matrix, word_by_topics: sp.csr_matrix, top_n: int = 10,
                   doc_word_pairs: list[tuple[int, int]] = None, weights: list[float] = None) -> tuple:
    if doc_word_pairs is None:
        doc_word_pairs = []
    if weights is None:
        weights = []

    topic_docs = []
    topic_doc_weights = []
    topic_words = []
    topic_word_weights = []

    if len(doc_word_pairs) > 0:
        for word_vec_id, doc_vec_id in doc_word_pairs:
            doc_vec = np.array(doc_by_topics[:, doc_vec_id].todense()).flatten()
            word_vec = np.array(word_by_topics[word_vec_id, :].todense()).flatten()
            doc_idxs = doc_vec.argsort()[-top_n:][::-1]
            word_idxs = word_vec.argsort()[-top_n:][::-1]
            topic_docs.append(doc_idxs)
            topic_doc_weights.append(doc_vec[doc_idxs])
            topic_words.append(word_idxs)
            topic_word_weights.append(word_vec[word_idxs])
    else:
        for i in range(word_by_topics.shape[0]):
            doc_vec = np.array(doc_by_topics[:, i].todense()).flatten()
            word_vec = np.array(word_by_topics[i, :].todense()).flatten()
            doc_idxs = doc_vec.argsort()[-top_n:][::-1]
            word_idxs = word_vec.argsort()[-top_n:][::-1]
            topic_docs.append(doc_idxs)
            topic_doc_weights.append(doc_vec[doc_idxs])
            topic_words.append(word_idxs)
            topic_word_weights.append(word_vec[word_idxs])


    return topic_words, topic_word_weights, topic_docs, topic_doc_weights
