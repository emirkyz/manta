#!/usr/bin/env python3
"""
Extract Top Articles from JSON to CSV

This script extracts article information for the top 10 articles from each topic
in a JSON file and saves them to a new CSV by matching with a source CSV.
"""

import json
import sys
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("Error: duckdb is not installed. Please install it with: pip install duckdb")
    sys.exit(1)

# =============================================================================
# CONFIGURATION - Edit these paths as needed
# =============================================================================

INPUT_JSON = "/Users/emirkarayagiz/Work/nmf-standalone/tests/beyza_hoca_makale/TopicAnalysis/Output/beyza_hoca_veri_pnmf_bpe_12/beyza_hoca_veri_pnmf_bpe_12_top_docs.json"
INPUT_CSV = "/Users/emirkarayagiz/Work/nmf-standalone/beyza_hoca_veri.csv"
OUTPUT_CSV = "/Users/emirkarayagiz/Work/nmf-standalone/extracted_top_articles.csv"

# =============================================================================


def load_json_abstracts(json_path):
    """
    Load JSON file and extract all abstracts from all topics.

    Returns:
        list: List of tuples (abstract_text, topic_name, rank_in_topic, score)
    """
    print(f"Loading JSON from: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}")
        sys.exit(1)

    abstracts = []
    topic_count = 0

    for topic_name, articles in data.items():
        topic_count += 1
        rank = 1  # Rank within the topic (1 = highest)

        for article_key, article_value in articles.items():
            # Split abstract text from score (format: "abstract_text:score")
            # Use rsplit with maxsplit=1 to handle colons in the abstract text
            try:
                abstract_text, score = article_value.rsplit(':', 1)
                abstracts.append((
                    abstract_text.strip(),
                    topic_name,
                    rank,
                    float(score)
                ))
                rank += 1
            except ValueError:
                print(f"Warning: Could not parse article in {topic_name}, key {article_key}")
                continue

    print(f"Found {topic_count} topics with {len(abstracts)} total articles")
    return abstracts


def match_abstracts_with_csv(abstracts, csv_path, output_path):
    """
    Match abstracts with source CSV and write output CSV.

    Args:
        abstracts: List of tuples (abstract_text, topic_name, rank, score)
        csv_path: Path to source CSV file
        output_path: Path to output CSV file
    """
    print(f"\nMatching with source CSV: {csv_path}")

    # Check if source CSV exists
    if not Path(csv_path).exists():
        print(f"Error: Source CSV file not found at {csv_path}")
        sys.exit(1)

    # Connect to DuckDB (in-memory database)
    con = duckdb.connect()

    try:
        # Read source CSV into a DuckDB table
        # Using read_csv_auto for automatic type detection
        con.execute("""
            CREATE TABLE source_articles AS
            SELECT * FROM read_csv_auto(?)
        """, [csv_path])

        # Get total count of source articles
        total_source = con.execute("SELECT COUNT(*) FROM source_articles").fetchone()[0]
        print(f"Source CSV contains {total_source} articles")

        # Create a temporary table for abstracts to match with topic info
        con.execute("""
            CREATE TABLE abstracts_to_match (
                abstract TEXT,
                topic TEXT,
                rank INTEGER,
                score DOUBLE
            )
        """)

        # Insert abstracts with their metadata
        for abstract_text, topic_name, rank, score in abstracts:
            con.execute(
                "INSERT INTO abstracts_to_match VALUES (?, ?, ?, ?)",
                [abstract_text, topic_name, rank, score]
            )

        print(f"\nSearching for {len(abstracts)} abstracts...")

        # Match abstracts with source articles and include topic/rank info
        # Using exact match on abstract text
        query = """
            SELECT DISTINCT
                a.topic,
                a.rank,
                s.*
            FROM source_articles s
            INNER JOIN abstracts_to_match a ON s.abstract = a.abstract
            ORDER BY a.topic, a.rank
        """

        matched_count = con.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0]
        print(f"Matched {matched_count} out of {len(abstracts)} articles")

        # Write output CSV
        print(f"\nWriting output to: {output_path}")
        con.execute(f"""
            COPY ({query})
            TO ?
            WITH (HEADER TRUE, DELIMITER ',')
        """, [output_path])

        # Report unmatched abstracts if any
        if matched_count < len(abstracts):
            print(f"\nWarning: {len(abstracts) - matched_count} abstracts could not be matched")

            # Find which abstracts didn't match
            unmatched_query = """
                SELECT a.topic, a.rank, a.abstract
                FROM abstracts_to_match a
                LEFT JOIN source_articles s ON a.abstract = s.abstract
                WHERE s.abstract IS NULL
                ORDER BY a.topic, a.rank
            """
            unmatched = con.execute(unmatched_query).fetchall()

            if len(unmatched) <= 5:
                print("\nUnmatched abstracts (first 200 chars):")
                for i, (topic, rank, abstract) in enumerate(unmatched, 1):
                    print(f"{i}. {topic} (Rank {rank}): {abstract[:200]}...")
            else:
                print(f"\nShowing first 5 unmatched abstracts (first 200 chars):")
                for i, (topic, rank, abstract) in enumerate(unmatched[:5], 1):
                    print(f"{i}. {topic} (Rank {rank}): {abstract[:200]}...")

        print(f"\nSuccess! Output saved to: {output_path}")
        print(f"Output CSV contains {matched_count} rows")

    except Exception as e:
        print(f"Error during matching: {e}")
        sys.exit(1)
    finally:
        con.close()


def main():
    """Main function"""
    print("=" * 70)
    print("Extract Top Articles from JSON to CSV")
    print("=" * 70)

    # Load abstracts from JSON
    abstracts = load_json_abstracts(INPUT_JSON)

    if not abstracts:
        print("Error: No abstracts found in JSON file")
        sys.exit(1)

    # Match with CSV and write output
    match_abstracts_with_csv(abstracts, INPUT_CSV, OUTPUT_CSV)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
