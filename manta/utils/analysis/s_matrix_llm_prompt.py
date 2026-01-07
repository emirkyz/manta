"""
S Matrix LLM Prompt Generator

Generates structured prompts for LLMs to interpret NMTF S matrix relationships.
Outputs both JSON (for API calls) and copy-paste text (for chat interfaces).

Usage:
    python -m manta.utils.analysis.s_matrix_llm_prompt \
        --word-scores output/analysis_word_scores.json \
        --top-docs output/analysis_top_docs.json \
        --s-matrix output/analysis_s_matrix.json \
        --coherence output/analysis_relevance_top_words.json \
        --output-json prompt.json \
        --output-txt prompt.txt
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


def load_json(path: str) -> dict:
    """Load JSON file and return contents."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_s_matrix(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load S matrix from JSON file.

    Returns:
        Tuple of (original_matrix, normalized_matrix)
    """
    data = load_json(path)

    # Handle new format with matrices dict
    if 'matrices' in data:
        original = np.array(data['matrices']['original'])
        normalized = np.array(data['matrices'].get('normalized', original))
    # Handle old format with just s_matrix
    elif 's_matrix' in data:
        original = np.array(data['s_matrix'])
        # Normalize columns to sum to 1
        col_sums = original.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        normalized = original / col_sums
    else:
        raise ValueError(f"Unknown S matrix format in {path}")

    return original, normalized


def analyze_s_matrix(
    s_matrix: np.ndarray,
    pure_threshold: float = 0.5,
    connection_threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Analyze S matrix to identify topic relationships.

    Args:
        s_matrix: Normalized S matrix (columns sum to 1)
        pure_threshold: Threshold for diagonal to be considered "pure topic"
        connection_threshold: Threshold for off-diagonal to be considered significant

    Returns:
        Dictionary with relationship analysis
    """
    n_topics = s_matrix.shape[0]

    # Analyze diagonal (self-connections = pure topics)
    diagonal = np.diag(s_matrix)
    pure_topics = [i + 1 for i in range(n_topics) if diagonal[i] >= pure_threshold]

    # Analyze off-diagonal connections
    strong_connections = []
    for i in range(n_topics):
        for j in range(n_topics):
            if i != j and s_matrix[i, j] >= connection_threshold:
                strong_connections.append({
                    "from": i + 1,
                    "to": j + 1,
                    "strength": round(float(s_matrix[i, j]), 4)
                })

    # Sort by strength
    strong_connections.sort(key=lambda x: x["strength"], reverse=True)

    # Identify hub topics (rows with multiple high outgoing connections)
    hub_topics = []
    for i in range(n_topics):
        outgoing = sum(1 for j in range(n_topics) if i != j and s_matrix[i, j] >= connection_threshold)
        if outgoing >= 2:
            hub_topics.append({"topic": i + 1, "influences_count": outgoing})
    hub_topics.sort(key=lambda x: x["influences_count"], reverse=True)

    # Identify sink topics (columns with multiple high incoming connections)
    sink_topics = []
    for j in range(n_topics):
        incoming = sum(1 for i in range(n_topics) if i != j and s_matrix[i, j] >= connection_threshold)
        if incoming >= 2:
            sink_topics.append({"topic": j + 1, "influenced_by_count": incoming})
    sink_topics.sort(key=lambda x: x["influenced_by_count"], reverse=True)

    return {
        "pure_topics": pure_topics,
        "hub_topics": hub_topics,
        "sink_topics": sink_topics,
        "strong_connections": strong_connections,
        "diagonal_values": [round(float(d), 4) for d in diagonal]
    }


def get_topic_influences(
    s_matrix: np.ndarray,
    topic_idx: int,
    threshold: float = 0.05
) -> Tuple[List[Dict], List[Dict]]:
    """
    Get topics that a given topic influences and is influenced by.

    Args:
        s_matrix: Normalized S matrix
        topic_idx: 0-based topic index
        threshold: Minimum strength to include

    Returns:
        Tuple of (influences, influenced_by) lists
    """
    n_topics = s_matrix.shape[0]

    # Topics this one influences (row values, excluding diagonal)
    influences = []
    for j in range(n_topics):
        if j != topic_idx and s_matrix[topic_idx, j] >= threshold:
            influences.append({
                "topic": j + 1,
                "strength": round(float(s_matrix[topic_idx, j]), 4)
            })
    influences.sort(key=lambda x: x["strength"], reverse=True)

    # Topics that influence this one (column values, excluding diagonal)
    influenced_by = []
    for i in range(n_topics):
        if i != topic_idx and s_matrix[i, topic_idx] >= threshold:
            influenced_by.append({
                "topic": i + 1,
                "strength": round(float(s_matrix[i, topic_idx]), 4)
            })
    influenced_by.sort(key=lambda x: x["strength"], reverse=True)

    return influences, influenced_by


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


def generate_s_matrix_prompt(
    word_scores_path: str,
    top_docs_path: str,
    s_matrix_path: str,
    coherence_path: Optional[str] = None,
    temporal_csv_path: Optional[str] = None,
    top_n_words: int = 10,
    top_n_docs: int = 3,
    pure_threshold: float = 0.5,
    connection_threshold: float = 0.1
) -> Tuple[Dict[str, Any], str]:
    """
    Generate structured JSON prompt and copy-paste text for LLM interpretation.

    Args:
        word_scores_path: Path to word scores JSON
        top_docs_path: Path to top docs JSON
        s_matrix_path: Path to S matrix JSON
        coherence_path: Optional path to coherence scores JSON
        temporal_csv_path: Optional path to temporal distribution CSV
        top_n_words: Number of top words to include per topic
        top_n_docs: Number of sample documents per topic
        pure_threshold: Threshold for pure topic identification
        connection_threshold: Threshold for significant connections

    Returns:
        Tuple of (json_prompt, text_prompt)
    """
    # Load all data
    word_scores = load_json(word_scores_path)
    top_docs = load_json(top_docs_path)
    _, s_matrix_normalized = load_s_matrix(s_matrix_path)

    coherence_data = None
    if coherence_path:
        coherence_data = load_json(coherence_path)

    # Load temporal data if available
    temporal_analysis = None
    if temporal_csv_path:
        temporal_df = load_temporal_csv(temporal_csv_path)
        temporal_analysis = analyze_temporal_trends(temporal_df)

    # Analyze S matrix
    s_analysis = analyze_s_matrix(
        s_matrix_normalized,
        pure_threshold=pure_threshold,
        connection_threshold=connection_threshold
    )

    # Parse documents
    parsed_docs = parse_top_docs(top_docs, top_n_docs)

    # Build topics list
    topics = []
    n_topics = s_matrix_normalized.shape[0]

    for topic_key, words_dict in word_scores.items():
        # Extract topic number
        topic_num = int(''.join(filter(str.isdigit, topic_key)))
        if topic_num > n_topics:
            continue

        topic_idx = topic_num - 1  # 0-based index

        # Get top words
        sorted_words = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)[:top_n_words]
        top_words = [w[0] for w in sorted_words]
        word_scores_list = [round(w[1], 4) for w in sorted_words]

        # Get coherence score if available
        coherence_score = None
        if coherence_data and 'gensim' in coherence_data:
            cv_per_topic = coherence_data['gensim'].get('c_v_per_topic', {})
            topic_coh_key = f"Topic {topic_num}"
            if topic_coh_key in cv_per_topic:
                coherence_score = round(cv_per_topic[topic_coh_key], 4)

        # Get sample docs
        sample_docs = parsed_docs.get(topic_num, [])

        # Get S matrix analysis for this topic
        influences, influenced_by = get_topic_influences(
            s_matrix_normalized, topic_idx, threshold=connection_threshold
        )

        self_connection = round(float(s_matrix_normalized[topic_idx, topic_idx]), 4)
        is_pure = topic_num in s_analysis["pure_topics"]

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
            "top_words": top_words,
            "word_scores": word_scores_list,
            "sample_docs": sample_docs,
            "coherence_score": coherence_score,
            "s_matrix_analysis": {
                "self_connection": self_connection,
                "is_pure_topic": is_pure,
                "influences": influences[:5],  # Limit to top 5
                "influenced_by": influenced_by[:5]
            }
        }
        if temporal_data:
            topic_entry["temporal_data"] = temporal_data

        topics.append(topic_entry)

    # Sort topics by ID
    topics.sort(key=lambda x: x["id"])

    # Build interpretation guide
    interpretation_guide = {
        "s_matrix": "Coupling matrix (k x k) showing how topic clusters relate to each other",
        "diagonal_values": "Self-connection strength - high values indicate 'pure' self-contained topics",
        "off_diagonal": "Cross-topic influence - S[i,j] means topic i contributes to/influences topic j",
        "normalized": "Values are normalized so each column sums to 1.0 (proportions of incoming influence)",
        "pure_topic": f"A topic with self-connection >= {pure_threshold} (mostly self-contained)",
        "hub_topic": "A topic that influences multiple other topics",
        "sink_topic": "A topic that is influenced by multiple other topics"
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
        "Explain what real-world concepts or relationships might cause the strong cross-topic connections",
        "Why might certain topics be 'pure' (self-contained) while others are highly connected?",
        "What overarching themes or categories do you see across the topics?",
        "Are there any topic clusters (groups of mutually related topics)?"
    ]

    # Add temporal questions if data available
    if temporal_analysis:
        questions.extend([
            "Based on the temporal trends, which topics are emerging (growing in importance)?",
            "Which topics appear to be declining in research interest?",
            "Do any spikes or trend changes correlate with real-world events (e.g., COVID-19 pandemic, policy changes, new technologies)?",
            "How do temporal trends relate to the S matrix relationships? Do hub topics show different temporal patterns than sink topics?"
        ])

    # Build output format
    output_format = {
        "type": "markdown",
        "topic_labels": "Dictionary mapping topic ID to descriptive label",
        "relationship_explanations": "List of explanations for strong connections",
        "pure_topic_analysis": "Why these topics are self-contained",
        "themes": "Overarching themes across all topics",
        "insights": "Key findings and recommendations"
    }

    # Add temporal output format if data available
    if temporal_analysis:
        output_format["temporal_analysis"] = {
            "emerging_topics": "List of topics showing growth with explanations",
            "declining_topics": "List of topics showing decline with explanations",
            "event_correlations": "Connections between temporal patterns and real-world events",
            "temporal_relationship_insights": "How temporal trends relate to S matrix relationships"
        }

    # Build JSON prompt
    json_prompt = {
        "task": "Analyze topic relationships in this NMTF topic model and provide interpretations",
        "context": {
            "model_type": "NMTF (Non-negative Matrix Tri-Factorization)",
            "num_topics": n_topics,
            "has_temporal_data": temporal_analysis is not None,
            "interpretation_guide": interpretation_guide
        },
        "topics": topics,
        "relationships": {
            "pure_topics": s_analysis["pure_topics"],
            "hub_topics": s_analysis["hub_topics"],
            "sink_topics": s_analysis["sink_topics"],
            "strong_connections": s_analysis["strong_connections"][:15],  # Top 15
            "s_matrix_diagonal": s_analysis["diagonal_values"]
        },
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
        "You are analyzing an NMTF (Non-negative Matrix Tri-Factorization) topic model.",
        "The S matrix shows relationships between topics - diagonal values indicate 'pure' topics,",
        "while off-diagonal values show cross-topic influence.",
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

        # S matrix info
        s_info = topic["s_matrix_analysis"]
        pure_str = " [PURE TOPIC]" if s_info["is_pure_topic"] else ""
        text_lines.append(f"  Self-connection: {s_info['self_connection']}{pure_str}")

        if s_info["influences"]:
            inf_str = ", ".join([f"T{i['topic']}({i['strength']})" for i in s_info["influences"][:3]])
            text_lines.append(f"  Influences: {inf_str}")

        if s_info["influenced_by"]:
            inf_by_str = ", ".join([f"T{i['topic']}({i['strength']})" for i in s_info["influenced_by"][:3]])
            text_lines.append(f"  Influenced by: {inf_by_str}")

        # Add temporal info if available
        if "temporal_data" in topic:
            t_data = topic["temporal_data"]
            trend_symbol = {"emerging": "↑", "declining": "↓", "stable": "→", "volatile": "~"}.get(t_data["trend"], "?")
            growth_pct = t_data["avg_growth_rate"] * 100
            text_lines.append(f"  Trend: {t_data['trend'].upper()} {trend_symbol} ({growth_pct:+.1f}% avg) | Peak: {t_data['peak_period']} ({t_data['peak_count']} docs)")

        if topic["sample_docs"]:
            text_lines.append(f"  Sample: \"{topic['sample_docs'][0][:100]}...\"")

        text_lines.append("")

    # Relationships summary
    text_lines.extend([
        "=" * 60,
        "S MATRIX RELATIONSHIPS",
        "=" * 60,
        ""
    ])

    if s_analysis["pure_topics"]:
        text_lines.append(f"Pure topics (self-contained): {s_analysis['pure_topics']}")

    if s_analysis["hub_topics"]:
        hub_str = ", ".join([f"T{h['topic']}(influences {h['influences_count']})" for h in s_analysis["hub_topics"]])
        text_lines.append(f"Hub topics (influence many): {hub_str}")

    if s_analysis["sink_topics"]:
        sink_str = ", ".join([f"T{s['topic']}(from {s['influenced_by_count']})" for s in s_analysis["sink_topics"]])
        text_lines.append(f"Sink topics (influenced by many): {sink_str}")

    text_lines.append("")
    text_lines.append("Strong connections:")
    for conn in s_analysis["strong_connections"][:10]:
        text_lines.append(f"  Topic {conn['from']} -> Topic {conn['to']}: {conn['strength']}")

    # Add temporal analysis section if available
    if temporal_analysis:
        summary = temporal_analysis["summary"]
        text_lines.extend([
            "",
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

    # Task section
    task_items = [
        "1. **Topic Labels**: Suggest a descriptive name for each topic based on its words and documents",
        "2. **Relationship Explanations**: Why do certain topics influence others? What connects them?",
        "3. **Pure Topic Analysis**: Why are some topics self-contained while others are connected?",
        "4. **Themes**: What overarching themes do you see?",
        "5. **Insights**: Key findings about this topic structure"
    ]

    if temporal_analysis:
        task_items.extend([
            "6. **Temporal Trends**: Which topics are emerging vs declining? Why?",
            "7. **Event Correlations**: Do any spikes correlate with real-world events (COVID-19, policy changes, etc.)?",
            "8. **Temporal-Relationship Insights**: How do temporal trends relate to S matrix relationships?"
        ])

    text_lines.extend([
        "",
        "=" * 60,
        "YOUR TASK",
        "=" * 60,
        "",
        "Please analyze this topic model and provide:",
        ""
    ])
    text_lines.extend(task_items)

    # Output format
    output_keys = '"topic_labels": {1: "...", 2: "...", ...}, "relationships": [...], "themes": [...], "insights": [...]'
    if temporal_analysis:
        output_keys += ', "temporal_analysis": {"emerging": [...], "declining": [...], "event_correlations": [...], "temporal_relationship_insights": [...]}'

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
        description="Generate LLM prompts for S matrix interpretation"
    )
    parser.add_argument(
        "--word-scores", required=True,
        help="Path to word scores JSON file"
    )
    parser.add_argument(
        "--top-docs", required=True,
        help="Path to top docs JSON file"
    )
    parser.add_argument(
        "--s-matrix", required=True,
        help="Path to S matrix JSON file"
    )
    parser.add_argument(
        "--coherence", default=None,
        help="Path to coherence scores JSON file (optional)"
    )
    parser.add_argument(
        "--temporal-csv", default=None,
        help="Path to temporal distribution CSV file (optional, enables temporal analysis)"
    )
    parser.add_argument(
        "--output-json", default="s_matrix_prompt.json",
        help="Output path for JSON prompt"
    )
    parser.add_argument(
        "--output-txt", default="s_matrix_prompt.txt",
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
    parser.add_argument(
        "--pure-threshold", type=float, default=0.5,
        help="Threshold for pure topic identification (default: 0.5)"
    )
    parser.add_argument(
        "--connection-threshold", type=float, default=0.1,
        help="Threshold for significant connections (default: 0.1)"
    )

    args = parser.parse_args()

    print(f"Loading data...")
    print(f"  Word scores: {args.word_scores}")
    print(f"  Top docs: {args.top_docs}")
    print(f"  S matrix: {args.s_matrix}")
    if args.coherence:
        print(f"  Coherence: {args.coherence}")
    if args.temporal_csv:
        print(f"  Temporal CSV: {args.temporal_csv}")

    json_prompt, text_prompt = generate_s_matrix_prompt(
        word_scores_path=args.word_scores,
        top_docs_path=args.top_docs,
        s_matrix_path=args.s_matrix,
        coherence_path=args.coherence,
        temporal_csv_path=args.temporal_csv,
        top_n_words=args.top_n_words,
        top_n_docs=args.top_n_docs,
        pure_threshold=args.pure_threshold,
        connection_threshold=args.connection_threshold
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
    n_pure = len(json_prompt["relationships"]["pure_topics"])
    n_connections = len(json_prompt["relationships"]["strong_connections"])

    print(f"\nSummary:")
    print(f"  Topics: {n_topics}")
    print(f"  Pure topics: {n_pure}")
    print(f"  Strong connections: {n_connections}")
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
        # Edit these paths:
        WORD_SCORES_PATH = "/Users/emirkarayagiz/Downloads/cardiology_full_filtered_nmtf_bpe_47/cardiology_full_filtered_nmtf_bpe_47_word_scores.json"
        TOP_DOCS_PATH = "/Users/emirkarayagiz/Downloads/cardiology_full_filtered_nmtf_bpe_47/cardiology_full_filtered_nmtf_bpe_47_top_docs.json"
        S_MATRIX_PATH = "/Users/emirkarayagiz/Downloads/cardiology_full_filtered_nmtf_bpe_47/cardiology_full_filtered_nmtf_bpe_47_s_matrix.json"
        COHERENCE_PATH = "/Users/emirkarayagiz/Downloads/cardiology_full_filtered_nmtf_bpe_47/cardiology_full_filtered_nmtf_bpe_47_relevance_top_words.json"  # Optional, set to None if not available

        # Optional: Path to temporal distribution CSV (set to None to disable temporal analysis)
        TEMPORAL_CSV_PATH = "/Users/emirkarayagiz/Downloads/cardiology_full_filtered_nmtf_bpe_47/cardiology_full_filtered_nmtf_bpe_47_temporal_topic_dist_quarter.csv"

        # Output paths:
        OUTPUT_JSON = "s_matrix_prompt.json"
        OUTPUT_TXT = "s_matrix_prompt.txt"

        # Settings:
        TOP_N_WORDS = 15
        TOP_N_DOCS = 15
        PURE_THRESHOLD = 0.5
        CONNECTION_THRESHOLD = 0.1

        # Run
        print(f"Loading data...")
        print(f"  Word scores: {WORD_SCORES_PATH}")
        print(f"  Top docs: {TOP_DOCS_PATH}")
        print(f"  S matrix: {S_MATRIX_PATH}")
        if COHERENCE_PATH:
            print(f"  Coherence: {COHERENCE_PATH}")
        if TEMPORAL_CSV_PATH:
            print(f"  Temporal CSV: {TEMPORAL_CSV_PATH}")

        json_prompt, text_prompt = generate_s_matrix_prompt(
            word_scores_path=WORD_SCORES_PATH,
            top_docs_path=TOP_DOCS_PATH,
            s_matrix_path=S_MATRIX_PATH,
            coherence_path=COHERENCE_PATH,
            temporal_csv_path=TEMPORAL_CSV_PATH,
            top_n_words=TOP_N_WORDS,
            top_n_docs=TOP_N_DOCS,
            pure_threshold=PURE_THRESHOLD,
            connection_threshold=CONNECTION_THRESHOLD
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
        n_pure = len(json_prompt["relationships"]["pure_topics"])
        n_connections = len(json_prompt["relationships"]["strong_connections"])

        print(f"\nSummary:")
        print(f"  Topics: {n_topics}")
        print(f"  Pure topics: {n_pure}")
        print(f"  Strong connections: {n_connections}")
        if json_prompt['context'].get('has_temporal_data'):
            summary = json_prompt.get('temporal_summary', {})
            print(f"  Temporal range: {summary.get('period_range', ['?', '?'])[0]} to {summary.get('period_range', ['?', '?'])[1]}")
            print(f"  Emerging topics: {summary.get('emerging_topics', [])}")
            print(f"  Declining topics: {summary.get('declining_topics', [])}")
        print(f"\nCopy the content of {OUTPUT_TXT} to paste into ChatGPT/Claude")

    else:
        # CLI mode
        main()