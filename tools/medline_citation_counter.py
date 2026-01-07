#!/usr/bin/env python3
"""
Parallel version of citation extractor using ProcessPoolExecutor.
Respects NCBI rate limits while processing multiple PMIDs concurrently.
Saves results incrementally to prevent data loss.
"""

import subprocess
import csv
import re
import sys
import time
from typing import List, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


# Rate limiting parameters (requests per second)
RATE_LIMIT = 3.0


def extract_pmids_from_medline(file_path: str) -> List[str]:
    """Extract PMIDs from MEDLINE formatted text file."""
    pmids = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('PMID-'):
                    pmid = line.replace('PMID-', '').strip()
                    pmids.append(pmid)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    return pmids


def get_citing_articles(pmid: str, rate_limit: float = 3.0) -> Set[str]:
    """Get articles that cite the given PMID."""
    time.sleep(1.0 / rate_limit)  # Simple rate limiting
    try:
        result = subprocess.run(
            ['elink', '-db', 'pubmed', '-id', pmid, '-cited'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            pmids = re.findall(r'\b\d{7,9}\b', result.stdout)
            return set(pmids)
        return set()
    except Exception as e:
        print(f"Warning: Error for PMID {pmid} (cited): {e}")
        return set()


def get_cited_articles(pmid: str, rate_limit: float = 3.0) -> Set[str]:
    """Get articles cited by the given PMID."""
    time.sleep(1.0 / rate_limit)  # Simple rate limiting
    try:
        result = subprocess.run(
            ['elink', '-db', 'pubmed', '-id', pmid, '-cites'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            pmids = re.findall(r'\b\d{7,9}\b', result.stdout)
            return set(pmids)
        return set()
    except Exception as e:
        print(f"Warning: Error for PMID {pmid} (cites): {e}")
        return set()


def process_pmid(args_tuple: Tuple[str, float]) -> Tuple[str, Set[str], Set[str]]:
    """Process a single PMID and return citation data."""
    pmid, rate_limit = args_tuple
    citing = get_citing_articles(pmid, rate_limit)
    cited = get_cited_articles(pmid, rate_limit)
    return pmid, citing, cited


def main():
    parser = argparse.ArgumentParser(
        description='Extract citation data from MEDLINE file (parallel version)'
    )
    parser.add_argument('input_file', help='Input MEDLINE formatted text file')
    parser.add_argument('output_file', nargs='?', default='citations_parallel.csv',
                        help='Output CSV file (default: citations_parallel.csv)')
    parser.add_argument('-w', '--workers', type=int, default=5,
                        help='Number of parallel workers (default: 5)')
    parser.add_argument('-r', '--rate', type=float, default=3.0,
                        help='Max requests per second (default: 3.0, use 10.0 with API key)')
    parser.add_argument('-s', '--save-interval', type=int, default=100,
                        help='Flush results to disk every N records (default: 100)')

    args = parser.parse_args()
    args.rate = 20.0
    print(f"Reading PMIDs from: {args.input_file}")
    pmids = extract_pmids_from_medline(args.input_file)

    if not pmids:
        print("No PMIDs found in the file!")
        sys.exit(1)
    import os
    args.workers = os.cpu_count()

    print(f"Found {len(pmids)} PMIDs")
    print(f"Using {args.workers} parallel workers")
    print(f"Rate limit: {args.rate} requests/second")
    print(f"Save interval: every {args.save_interval} records")
    print(f"Fetching citation data...\n")

    completed = 0

    # Open CSV file and write header
    csvfile = open(args.output_file, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csvfile)
    writer.writerow(['Source_PMID', 'Cited_By_Count', 'Cites_Count'])
    csvfile.flush()

    try:
        # Process PMIDs in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks with rate limit parameter
            future_to_pmid = {
                executor.submit(process_pmid, (pmid, args.rate)): pmid
                for pmid in pmids
            }

            # Process completed tasks
            for future in as_completed(future_to_pmid):
                pmid = future_to_pmid[future]
                try:
                    pmid, citing, cited = future.result()

                    # Write result immediately
                    writer.writerow([
                        pmid,
                        len(citing),
                        len(cited)
                    ])

                    completed += 1

                    # Flush to disk every N records
                    if completed % args.save_interval == 0:
                        csvfile.flush()
                        print(f"[{completed}/{len(pmids)}] PMID {pmid}: "
                              f"cited_by={len(citing)}, cites={len(cited)} (saved)")
                    else:
                        print(f"[{completed}/{len(pmids)}] PMID {pmid}: "
                              f"cited_by={len(citing)}, cites={len(cited)}")

                except Exception as e:
                    print(f"Error processing PMID {pmid}: {e}")
                    completed += 1

        # Final flush
        csvfile.flush()

    finally:
        csvfile.close()

    print(f"\nDone! Processed {len(pmids)} PMIDs")
    print(f"Results saved to: {args.output_file}")


if __name__ == '__main__':
    main()