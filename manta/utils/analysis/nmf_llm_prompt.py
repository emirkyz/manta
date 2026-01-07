"""
NMF LLM Prompt Generator

Generates structured prompts for LLMs to interpret NMF topic models.
Outputs both JSON (for API calls) and copy-paste text (for chat interfaces).

This is a simplified version without S matrix analysis, suitable for standard NMF models.

Usage:
    python -m manta.utils.analysis.nmf_llm_prompt \
        --top-docs output/analysis_top_docs.json \
        --coherence output/analysis_relevance_top_words.json \
        --use-relevance-words \
        --output-json prompt.json \
        --output-txt prompt.txt

    # Or using word_scores.json instead of relevance:
    python -m manta.utils.analysis.nmf_llm_prompt \
        --word-scores output/analysis_word_scores.json \
        --top-docs output/analysis_top_docs.json \
        --output-json prompt.json \
        --output-txt prompt.txt
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


def load_json(path: str) -> dict:
    """Load JSON file and return contents."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_relevance_words(coherence_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load relevance-scored words from coherence JSON.

    Args:
        coherence_path: Path to coherence JSON file

    Returns:
        Dictionary mapping topic keys (e.g., 'topic_01') to word scores
    """
    data = load_json(coherence_path)
    relevance = data.get('relevance', {})
    if not relevance:
        raise ValueError(f"No 'relevance' section found in {coherence_path}")
    return relevance


def parse_top_docs(top_docs: Dict[str, Dict[str, str]], top_n: int = 10) -> Dict[int, List[str]]:
    """
    Parse top docs JSON and extract sample documents.

    Args:
        top_docs: Top docs dictionary from JSON
        top_n: Number of documents to include per topic

    Returns:
        Dictionary mapping topic number to list of document excerpts
    """
    result = {}

    for topic_key, docs in top_docs.items():
        # Extract topic number from key like "Topic 1" or "Topic 01"
        topic_num = int(''.join(filter(str.isdigit, topic_key)))

        # Get documents sorted by score (embedded in the value string)
        doc_list = []
        for doc_id, doc_text_score in list(docs.items())[:top_n]:
            # Split by last colon to get text and score
            if ':' in doc_text_score:
                parts = doc_text_score.rsplit(':', 1)
                doc_text = parts[0]
            else:
                doc_text = doc_text_score

            # Truncate long documents
            if len(doc_text) > 200:
                doc_text = doc_text[:200] + "..."

            doc_list.append(doc_text)

        result[topic_num] = doc_list

    return result


def get_words_from_relevance(
    relevance_data: Dict[str, Dict[str, float]],
    top_n_words: int = 10
) -> Dict[int, Tuple[List[str], List[float]]]:
    """
    Extract top words from relevance data.

    Args:
        relevance_data: Relevance section from coherence JSON
        top_n_words: Number of top words per topic

    Returns:
        Dictionary mapping topic number to (words, scores) tuple
    """
    result = {}

    for topic_key, words_dict in relevance_data.items():
        # Extract topic number from key like "topic_01" or "Topic 1"
        topic_num = int(''.join(filter(str.isdigit, topic_key)))

        # Sort by score (descending - less negative is better for relevance)
        sorted_words = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)[:top_n_words]
        words = [w[0] for w in sorted_words]
        scores = [round(w[1], 4) for w in sorted_words]

        result[topic_num] = (words, scores)

    return result


def get_words_from_word_scores(
    word_scores: Dict[str, Dict[str, float]],
    top_n_words: int = 10
) -> Dict[int, Tuple[List[str], List[float]]]:
    """
    Extract top words from word_scores.json.

    Args:
        word_scores: Word scores dictionary from JSON
        top_n_words: Number of top words per topic

    Returns:
        Dictionary mapping topic number to (words, scores) tuple
    """
    result = {}

    for topic_key, words_dict in word_scores.items():
        # Extract topic number from key like "Topic 1" or "Topic 01"
        topic_num = int(''.join(filter(str.isdigit, topic_key)))

        # Sort by score (descending)
        sorted_words = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)[:top_n_words]
        words = [w[0] for w in sorted_words]
        scores = [round(w[1], 4) for w in sorted_words]

        result[topic_num] = (words, scores)

    return result


def load_temporal_csv(path: str) -> pd.DataFrame:
    """
    Load temporal distribution CSV file.

    Args:
        path: Path to temporal CSV file

    Returns:
        DataFrame with period as index and topic columns
    """
    df = pd.read_csv(path)
    df = df.set_index('period')
    return df


def analyze_temporal_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze temporal trends for each topic.

    Args:
        df: DataFrame with period as index and topic columns

    Returns:
        Dictionary with trend analysis per topic and summary
    """
    topic_trends = {}
    periods = list(df.index)

    for col in df.columns:
        # Extract topic number from column name like "Topic 1"
        topic_num = int(''.join(filter(str.isdigit, col)))
        counts = df[col].values
        counts_dict = {str(p): int(c) for p, c in zip(periods, counts)}

        # Calculate year-over-year changes
        changes = []
        for i in range(1, len(counts)):
            if counts[i-1] > 0:
                pct_change = (counts[i] - counts[i-1]) / counts[i-1]
                changes.append(pct_change)
            elif counts[i] > 0:
                changes.append(1.0)  # From 0 to positive = 100% growth
            else:
                changes.append(0.0)

        # Determine trend
        if len(changes) >= 2:
            avg_change = np.mean(changes)
            recent_changes = changes[-3:] if len(changes) >= 3 else changes

            # Check for consistent direction
            positive_changes = sum(1 for c in recent_changes if c > 0.1)
            negative_changes = sum(1 for c in recent_changes if c < -0.1)

            if positive_changes >= len(recent_changes) * 0.6 and avg_change > 0.1:
                trend = "emerging"
            elif negative_changes >= len(recent_changes) * 0.6 and avg_change < -0.1:
                trend = "declining"
            elif abs(avg_change) < 0.15:
                trend = "stable"
            else:
                trend = "volatile"
        else:
            trend = "insufficient_data"
            avg_change = 0.0

        # Find peak period
        peak_idx = np.argmax(counts)
        peak_period = str(periods[peak_idx])
        peak_count = int(counts[peak_idx])

        # Find total count
        total_count = int(np.sum(counts))

        topic_trends[topic_num] = {
            "counts_by_period": counts_dict,
            "trend": trend,
            "peak_period": peak_period,
            "peak_count": peak_count,
            "total_count": total_count,
            "avg_growth_rate": round(float(avg_change), 4) if changes else 0.0
        }

    # Build summary
    emerging = [t for t, data in topic_trends.items() if data["trend"] == "emerging"]
    declining = [t for t, data in topic_trends.items() if data["trend"] == "declining"]
    stable = [t for t, data in topic_trends.items() if data["trend"] == "stable"]

    # Detect notable spikes (periods with unusually high counts)
    notable_events = []
    period_totals = df.sum(axis=1)
    mean_total = period_totals.mean()
    std_total = period_totals.std()

    for period, total in period_totals.items():
        if total > mean_total + 1.5 * std_total:
            # Find which topics spiked
            period_data = df.loc[period]
            topic_means = df.mean()
            spiked_topics = []
            for col in df.columns:
                topic_num = int(''.join(filter(str.isdigit, col)))
                if period_data[col] > topic_means[col] * 1.5:
                    spiked_topics.append(topic_num)
            if spiked_topics:
                notable_events.append({
                    "period": str(period),
                    "total_count": int(total),
                    "spiked_topics": spiked_topics,
                    "description": f"Significant spike in Topics {', '.join(map(str, spiked_topics[:5]))}"
                })

    summary = {
        "period_range": [str(periods[0]), str(periods[-1])],
        "num_periods": len(periods),
        "emerging_topics": sorted(emerging),
        "declining_topics": sorted(declining),
        "stable_topics": sorted(stable),
        "notable_events": notable_events[:5]  # Limit to top 5 events
    }

    return {
        "topic_trends": topic_trends,
        "summary": summary
    }


def generate_nmf_prompt(
    top_docs_path: str,
    word_scores_path: Optional[str] = None,
    coherence_path: Optional[str] = None,
    temporal_csv_path: Optional[str] = None,
    use_relevance_words: bool = False,
    top_n_words: int = 10,
    top_n_docs: int = 3
) -> Tuple[Dict[str, Any], str]:
    """
    Generate structured JSON prompt and copy-paste text for LLM interpretation.

    Args:
        top_docs_path: Path to top docs JSON
        word_scores_path: Path to word scores JSON (required if use_relevance_words=False)
        coherence_path: Path to coherence scores JSON (required if use_relevance_words=True)
        temporal_csv_path: Path to temporal distribution CSV (optional)
        use_relevance_words: If True, use words from coherence['relevance'], else from word_scores.json
        top_n_words: Number of top words to include per topic
        top_n_docs: Number of sample documents per topic

    Returns:
        Tuple of (json_prompt, text_prompt)
    """
    # Validate inputs
    if use_relevance_words:
        if not coherence_path:
            raise ValueError("--coherence is required when --use-relevance-words is True")
    else:
        if not word_scores_path:
            raise ValueError("--word-scores is required when --use-relevance-words is False")

    # Load top docs
    top_docs = load_json(top_docs_path)
    parsed_docs = parse_top_docs(top_docs, top_n_docs)

    # Load words based on source
    if use_relevance_words:
        relevance_data = load_relevance_words(coherence_path)
        topic_words = get_words_from_relevance(relevance_data, top_n_words)
        word_source = "relevance"
    else:
        word_scores = load_json(word_scores_path)
        topic_words = get_words_from_word_scores(word_scores, top_n_words)
        word_source = "word_scores"

    # Load coherence scores if available
    coherence_data = None
    if coherence_path:
        coherence_data = load_json(coherence_path)

    # Load temporal data if available
    temporal_analysis = None
    if temporal_csv_path:
        temporal_df = load_temporal_csv(temporal_csv_path)
        temporal_analysis = analyze_temporal_trends(temporal_df)

    # Get number of topics
    n_topics = len(topic_words)

    # Build topics list
    topics = []
    for topic_num in sorted(topic_words.keys()):
        words, scores = topic_words[topic_num]

        # Get coherence score if available
        coherence_score = None
        if coherence_data and 'gensim' in coherence_data:
            cv_per_topic = coherence_data['gensim'].get('c_v_per_topic', {})
            topic_coh_key = f"Topic {topic_num}"
            if topic_coh_key in cv_per_topic:
                coherence_score = round(cv_per_topic[topic_coh_key], 4)

        # Get sample docs
        sample_docs = parsed_docs.get(topic_num, [])

        # Get temporal data if available
        temporal_data = None
        if temporal_analysis and topic_num in temporal_analysis["topic_trends"]:
            t_data = temporal_analysis["topic_trends"][topic_num]
            temporal_data = {
                "counts_by_period": t_data["counts_by_period"],
                "trend": t_data["trend"],
                "peak_period": t_data["peak_period"],
                "peak_count": t_data["peak_count"],
                "total_count": t_data["total_count"],
                "avg_growth_rate": t_data["avg_growth_rate"]
            }

        topic_entry = {
            "id": topic_num,
            "top_words": words,
            "word_scores": scores,
            "sample_docs": sample_docs,
            "coherence_score": coherence_score
        }
        if temporal_data:
            topic_entry["temporal_data"] = temporal_data

        topics.append(topic_entry)

    # Build JSON prompt
    interpretation_guide = {
        "top_words": "Most important words for each topic, ranked by score",
        "word_scores": "Numeric scores indicating word importance (higher = more important)",
        "sample_docs": "Representative documents strongly associated with each topic",
        "coherence_score": "Topic coherence score (higher = more semantically coherent)"
    }

    # Add temporal interpretation guide if available
    if temporal_analysis:
        interpretation_guide["temporal_data"] = {
            "counts_by_period": "Number of documents per time period",
            "trend": "Overall trend: emerging (growing), declining (shrinking), stable, or volatile",
            "peak_period": "Time period with highest document count",
            "avg_growth_rate": "Average period-over-period growth rate (positive = growth, negative = decline)"
        }

    # Build questions list
    questions = [
        "Based on the top words and sample documents, suggest a meaningful label/name for each topic",
        "What are the main themes or concepts represented by each topic?",
        "Are there any topics that seem related or overlapping?",
        "Which topics appear to be the most coherent and well-defined?",
        "Are there any topics that seem too broad or too specific?"
    ]

    # Add temporal questions if data available
    if temporal_analysis:
        questions.extend([
            "Based on the temporal trends, which topics are emerging (growing in importance)?",
            "Which topics appear to be declining in research interest?",
            "Do any spikes or trend changes correlate with real-world events (e.g., COVID-19 pandemic, policy changes, new technologies)?",
            "What future trends might you predict based on the temporal patterns?"
        ])

    # Build output format
    output_format = {
        "topic_labels": "Dictionary mapping topic ID to descriptive label",
        "topic_descriptions": "Brief description of what each topic represents",
        "related_topics": "Groups of topics that share common themes",
        "quality_assessment": "Assessment of topic quality and coherence",
        "insights": "Key findings and recommendations"
    }

    # Add temporal output format if data available
    if temporal_analysis:
        output_format["temporal_analysis"] = {
            "emerging_topics": "List of topics showing growth with explanations",
            "declining_topics": "List of topics showing decline with explanations",
            "event_correlations": "Connections between temporal patterns and real-world events",
            "future_predictions": "Predictions for topic trends"
        }

    json_prompt = {
        "task": "Analyze topics in this NMF topic model and provide interpretations",
        "context": {
            "model_type": "NMF (Non-negative Matrix Factorization)",
            "num_topics": n_topics,
            "word_source": word_source,
            "has_temporal_data": temporal_analysis is not None,
            "interpretation_guide": interpretation_guide
        },
        "topics": topics,
        "analysis_request": {
            "questions": questions,
            "output_format": output_format
        }
    }

    # Add temporal summary to JSON if available
    if temporal_analysis:
        json_prompt["temporal_summary"] = temporal_analysis["summary"]

    # Build copy-paste text prompt
    text_lines = [
        "You are analyzing an NMF (Non-negative Matrix Factorization) topic model.",
        f"Words are sourced from: {word_source}",
    ]

    if temporal_analysis:
        summary = temporal_analysis["summary"]
        text_lines.append(f"Temporal data available: {summary['period_range'][0]} to {summary['period_range'][1]}")

    text_lines.extend([
        "",
        "=" * 60,
        "TOPICS OVERVIEW",
        "=" * 60,
        ""
    ])

    for topic in topics:
        words_str = ", ".join(topic["top_words"][:7])
        coh_str = f" (coherence: {topic['coherence_score']})" if topic['coherence_score'] else ""

        text_lines.append(f"**Topic {topic['id']}**: {words_str}{coh_str}")

        # Add temporal info if available
        if "temporal_data" in topic:
            t_data = topic["temporal_data"]
            trend_symbol = {"emerging": "↑", "declining": "↓", "stable": "→", "volatile": "~"}.get(t_data["trend"], "?")
            growth_pct = t_data["avg_growth_rate"] * 100
            text_lines.append(f"  Trend: {t_data['trend'].upper()} {trend_symbol} ({growth_pct:+.1f}% avg) | Peak: {t_data['peak_period']} ({t_data['peak_count']} docs)")

        if topic["sample_docs"]:
            text_lines.append(f"  Sample: \"{topic['sample_docs'][0][:100]}...\"")

        text_lines.append("")

    # Add temporal analysis section if available
    if temporal_analysis:
        summary = temporal_analysis["summary"]
        text_lines.extend([
            "=" * 60,
            "TEMPORAL ANALYSIS",
            "=" * 60,
            "",
            f"Time range: {summary['period_range'][0]} to {summary['period_range'][1]} ({summary['num_periods']} periods)",
            ""
        ])

        if summary["emerging_topics"]:
            text_lines.append(f"EMERGING topics (growing): {summary['emerging_topics']}")
        if summary["declining_topics"]:
            text_lines.append(f"DECLINING topics (shrinking): {summary['declining_topics']}")
        if summary["stable_topics"]:
            text_lines.append(f"STABLE topics: {summary['stable_topics']}")

        if summary["notable_events"]:
            text_lines.extend(["", "Notable spikes/events:"])
            for event in summary["notable_events"]:
                text_lines.append(f"  - {event['period']}: {event['description']} (total: {event['total_count']} docs)")

        text_lines.append("")

    # Task section
    task_items = [
        "1. **Topic Labels**: Suggest a descriptive name for each topic based on its words and documents",
        "2. **Topic Descriptions**: Brief description of what each topic represents",
        "3. **Related Topics**: Which topics share common themes?",
        "4. **Quality Assessment**: Which topics are most/least coherent?",
        "5. **Insights**: Key findings about this topic structure"
    ]

    if temporal_analysis:
        task_items.extend([
            "6. **Temporal Trends**: Which topics are emerging vs declining? Why?",
            "7. **Event Correlations**: Do any spikes correlate with real-world events (COVID-19, policy changes, etc.)?",
            "8. **Future Predictions**: What trends might continue based on the temporal patterns?"
        ])

    text_lines.extend([
        "=" * 60,
        "YOUR TASK",
        "=" * 60,
        "",
        "Please analyze this topic model and provide:",
        ""
    ])
    text_lines.extend(task_items)

    # Output format
    output_keys = '"topic_labels": {1: "...", 2: "...", ...}, "descriptions": {...}, "related": [...], "quality": {...}, "insights": [...]'
    if temporal_analysis:
        output_keys += ', "temporal_analysis": {"emerging": [...], "declining": [...], "event_correlations": [...], "predictions": [...]}'

    text_lines.extend([
        "",
        "Respond in JSON format with these keys:",
        '{' + output_keys + '}',
        ""
    ])

    text_prompt = "\n".join(text_lines)

    return json_prompt, text_prompt


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate LLM prompts for NMF topic interpretation"
    )
    parser.add_argument(
        "--word-scores", default=None,
        help="Path to word scores JSON file (required if not using --use-relevance-words)"
    )
    parser.add_argument(
        "--top-docs", required=True,
        help="Path to top docs JSON file"
    )
    parser.add_argument(
        "--coherence", default=None,
        help="Path to coherence scores JSON file (required if using --use-relevance-words)"
    )
    parser.add_argument(
        "--temporal-csv", default=None,
        help="Path to temporal distribution CSV file (optional, enables temporal analysis)"
    )
    parser.add_argument(
        "--use-relevance-words", action="store_true",
        help="Use relevance-scored words from coherence JSON instead of word_scores.json"
    )
    parser.add_argument(
        "--output-json", default="nmf_prompt.json",
        help="Output path for JSON prompt"
    )
    parser.add_argument(
        "--output-txt", default="nmf_prompt.txt",
        help="Output path for text prompt"
    )
    parser.add_argument(
        "--top-n-words", type=int, default=10,
        help="Number of top words per topic (default: 10)"
    )
    parser.add_argument(
        "--top-n-docs", type=int, default=3,
        help="Number of sample docs per topic (default: 3)"
    )

    args = parser.parse_args()

    print(f"Loading data...")
    print(f"  Top docs: {args.top_docs}")
    if args.use_relevance_words:
        print(f"  Word source: coherence relevance")
        print(f"  Coherence: {args.coherence}")
    else:
        print(f"  Word source: word_scores.json")
        print(f"  Word scores: {args.word_scores}")
    if args.temporal_csv:
        print(f"  Temporal CSV: {args.temporal_csv}")

    json_prompt, text_prompt = generate_nmf_prompt(
        top_docs_path=args.top_docs,
        word_scores_path=args.word_scores,
        coherence_path=args.coherence,
        temporal_csv_path=args.temporal_csv,
        use_relevance_words=args.use_relevance_words,
        top_n_words=args.top_n_words,
        top_n_docs=args.top_n_docs
    )

    # Save JSON prompt
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(json_prompt, f, indent=2, ensure_ascii=False)
    print(f"\nJSON prompt saved to: {args.output_json}")

    # Save text prompt
    with open(args.output_txt, 'w', encoding='utf-8') as f:
        f.write(text_prompt)
    print(f"Text prompt saved to: {args.output_txt}")

    # Print summary
    n_topics = len(json_prompt["topics"])
    print(f"\nSummary:")
    print(f"  Topics: {n_topics}")
    print(f"  Word source: {json_prompt['context']['word_source']}")
    if json_prompt['context'].get('has_temporal_data'):
        summary = json_prompt.get('temporal_summary', {})
        print(f"  Temporal range: {summary.get('period_range', ['?', '?'])[0]} to {summary.get('period_range', ['?', '?'])[1]}")
        print(f"  Emerging topics: {summary.get('emerging_topics', [])}")
        print(f"  Declining topics: {summary.get('declining_topics', [])}")
    print(f"\nCopy the content of {args.output_txt} to paste into ChatGPT/Claude")


if __name__ == "__main__":
    import sys

    # ============================================================
    # SIMPLE MODE: Just edit these paths and run the file directly
    # ============================================================
    USE_SIMPLE_MODE = True  # Set to False to use CLI arguments

    if USE_SIMPLE_MODE:
        # ========== CONFIGURATION ==========
        # Toggle between word sources:
        USE_RELEVANCE_WORDS = True  # True = coherence relevance, False = word_scores.json

        # Paths (edit these):
        TOP_DOCS_PATH = "/results/radiology_imaging_full_test_k17/TopicAnalysis/Output/radiology_imaging_pnmf_bpe_25/radiology_imaging_pnmf_bpe_25_top_docs.json"

        # Only used if USE_RELEVANCE_WORDS = False:
        WORD_SCORES_PATH = ""

        # Required if USE_RELEVANCE_WORDS = True, optional otherwise:
        COHERENCE_PATH = "/results/radiology_imaging_full_test_k17/TopicAnalysis/Output/radiology_imaging_pnmf_bpe_25/radiology_imaging_pnmf_bpe_25_coherence_scores.json"

        # Optional: Path to temporal distribution CSV (set to None to disable temporal analysis)
        TEMPORAL_CSV_PATH = "/results/radiology_imaging_full_test_k17/TopicAnalysis/Output/radiology_imaging_pnmf_bpe_25/radiology_imaging_pnmf_bpe_25_temporal_topic_dist_quarter.csv"

        # Output paths:
        OUTPUT_JSON = "nmf_prompt.json"
        OUTPUT_TXT = "nmf_prompt.txt"

        # Settings:
        TOP_N_WORDS = 15
        TOP_N_DOCS = 15
        # ===================================

        # Run
        print(f"Loading data...")
        print(f"  Top docs: {TOP_DOCS_PATH}")
        if USE_RELEVANCE_WORDS:
            print(f"  Word source: coherence relevance")
            print(f"  Coherence: {COHERENCE_PATH}")
        else:
            print(f"  Word source: word_scores.json")
            print(f"  Word scores: {WORD_SCORES_PATH}")
        if TEMPORAL_CSV_PATH:
            print(f"  Temporal CSV: {TEMPORAL_CSV_PATH}")

        json_prompt, text_prompt = generate_nmf_prompt(
            top_docs_path=TOP_DOCS_PATH,
            word_scores_path=WORD_SCORES_PATH if not USE_RELEVANCE_WORDS else None,
            coherence_path=COHERENCE_PATH,
            temporal_csv_path=TEMPORAL_CSV_PATH,
            use_relevance_words=USE_RELEVANCE_WORDS,
            top_n_words=TOP_N_WORDS,
            top_n_docs=TOP_N_DOCS
        )

        # Save outputs
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(json_prompt, f, indent=2, ensure_ascii=False)
        print(f"\nJSON prompt saved to: {OUTPUT_JSON}")

        with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
            f.write(text_prompt)
        print(f"Text prompt saved to: {OUTPUT_TXT}")

        # Print summary
        n_topics = len(json_prompt["topics"])
        print(f"\nSummary:")
        print(f"  Topics: {n_topics}")
        print(f"  Word source: {json_prompt['context']['word_source']}")
        if json_prompt['context'].get('has_temporal_data'):
            summary = json_prompt.get('temporal_summary', {})
            print(f"  Temporal range: {summary.get('period_range', ['?', '?'])[0]} to {summary.get('period_range', ['?', '?'])[1]}")
            print(f"  Emerging topics: {summary.get('emerging_topics', [])}")
            print(f"  Declining topics: {summary.get('declining_topics', [])}")
        print(f"\nCopy the content of {OUTPUT_TXT} to paste into ChatGPT/Claude")

    else:
        # CLI mode
        main()
