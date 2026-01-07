#!/usr/bin/env python3
"""
Test script for violin plot functions using real radiology imaging analysis data.

This script loads actual NMF analysis results and datetime metadata to test
both the static PNG and interactive HTML violin plot generation functions.

Usage:
    python -m manta.utils.visualization.test_violin_plots
    or
    uv run python -m manta.utils.visualization.test_violin_plots
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from create_interactive_violin import generate_interactive_violin_plot
from violin_plot import gen_violin_plot


def main():
    """Main test function."""
    print("=" * 70)
    print("Violin Plot Test Script")
    print("=" * 70)

    # Define paths to test data
    base_dir = Path(__file__).parent.parent.parent.parent  # nmf-standalone root
    test_dir = (
        base_dir
        / "tests"
        / "radiology_imaging_full_test_k17"
        / "TopicAnalysis"
        / "Output"
    )

    output_dir = test_dir / "radiology_imaging_articles_paper_pnmf_bpe_25"
    table_name = "radiology_imaging_articles_paper_pnmf_bpe_25"

    model_file = output_dir / f"{table_name}_model_components.npz"
    metadata_file = test_dir / "radiology_imaging_articles_paper_bpe_tfidf_metadata.npz"
    csv_file = output_dir / f"{table_name}_temporal_topic_dist_quarter.csv"

    # Verify files exist
    if not model_file.exists():
        print(f"✗ Model file not found: {model_file}")
        sys.exit(1)
    if not metadata_file.exists():
        print(f"✗ Metadata file not found: {metadata_file}")
        sys.exit(1)
    if not csv_file.exists():
        print(f"✗ CSV file not found: {csv_file}")
        sys.exit(1)

    print(f"\n✓ Found model file: {model_file.name}")
    print(f"✓ Found metadata file: {metadata_file.name}")
    print(f"✓ Found CSV file: {csv_file.name}")
    print(f"✓ Output directory: {output_dir}")

    # Load NMF matrices
    print("\n" + "-" * 70)
    print("Loading NMF matrices...")
    print("-" * 70)

    model_data = np.load(model_file, allow_pickle=True)
    W = model_data["W"]
    H = model_data["H"]

    print(f"✓ W matrix shape: {W.shape} (documents × topics)")
    print(f"✓ H matrix shape: {H.shape} (topics × vocabulary)")

    n_docs, n_topics = W.shape
    print(f"\n  Documents: {n_docs:,}")
    print(f"  Topics: {n_topics}")

    # Load datetime metadata
    print("\n" + "-" * 70)
    print("Loading datetime metadata...")
    print("-" * 70)

    metadata = np.load(metadata_file, allow_pickle=True)
    datetime_year = metadata["datetime_year"]
    datetime_month = metadata["datetime_month"]

    print(f"✓ Datetime year shape: {datetime_year.shape}")
    print(f"✓ Datetime month shape: {datetime_month.shape}")
    print(f"\n  Year range: {datetime_year.min()} - {datetime_year.max()}")
    print(f"  Month range: {datetime_month.min()} - {datetime_month.max()}")

    # Create pandas datetime series
    print("\n" + "-" * 70)
    print("Creating datetime series...")
    print("-" * 70)

    # Construct datetime from year and month (use day=1 for all)
    datetime_series = pd.to_datetime(
        {
            "year": datetime_year,
            "month": datetime_month,
            "day": 1,
        }
    )

    print(f"✓ Created datetime series with {len(datetime_series):,} entries")
    print(f"  Date range: {datetime_series.min()} to {datetime_series.max()}")
    print(f"  Data type: {datetime_series.dtype}")

    # Generate static violin plot
    print("\n" + "-" * 70)
    print("Generating static PNG violin plot...")
    print("-" * 70)

    try:
        static_output = gen_violin_plot(
            W=W,
            S_matrix=None,
            datetime_series=datetime_series,
            table_output_dir=output_dir,
            table_name="radiology_imaging_test",
        )

        print(f"✓ Generated: {static_output.name}")
        print(f"  File size: {static_output.stat().st_size / 1024:.1f} KB")
        print(f"  Full path: {static_output}")

    except Exception as e:
        print(f"✗ Failed to generate static violin plot: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Generate interactive HTML violin plot (uses CSV data)
    print("\n" + "-" * 70)
    print("Generating interactive HTML violin plot from CSV...")
    print("-" * 70)

    try:
        interactive_output = generate_interactive_violin_plot(
            W=W,
            S_matrix=None,
            datetime_series=datetime_series,
            table_output_dir=output_dir,
            table_name=table_name,
        )

        print(f"✓ Generated: {interactive_output.name}")
        print(f"  File size: {interactive_output.stat().st_size / 1024:.1f} KB")
        print(f"  Full path: {interactive_output}")

        # Verify HTML content
        with open(interactive_output, "r") as f:
            html_content = f.read()

        checks = [
            ("Plotly.js library", "plotly-2.27.0.min.js" in html_content),
            ("Embedded data", "VIOLIN_DATA" in html_content),
            ("Topics field", '"topics":' in html_content),
            ("Metadata field", '"metadata":' in html_content),
            ("Checkbox controls", "topic-checkboxes" in html_content),
            ("CSV period format", '"periods":' in html_content),
            ("Quarter data", 'Q1' in html_content or 'Q2' in html_content),
        ]

        print("\n  Content verification:")
        for name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"    {status} {name}")

        if not all(check[1] for check in checks):
            print("\n✗ Some HTML content checks failed!")
            sys.exit(1)

    except Exception as e:
        print(f"✗ Failed to generate interactive violin plot: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Final summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"✓ All tests passed!")
    print(f"\nGenerated files:")
    print(f"  1. Static PNG: {static_output.name}")
    print(f"  2. Interactive HTML: {interactive_output.name}")
    print(f"\nYou can:")
    print(f"  - View the PNG: open {static_output}")
    print(f"  - Open the HTML: open {interactive_output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
