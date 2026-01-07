import csv
import re


def convert_month_to_numeric(month_str):
    """
    Convert month name/abbreviation to numeric format (1-12).
    Returns empty string if month cannot be mapped.
    """
    month_map = {
        # Standard abbreviations
        'jan': '1', 'feb': '2', 'mar': '3', 'apr': '4',
        'may': '5', 'jun': '6', 'jul': '7', 'aug': '8',
        'sep': '9', 'oct': '10', 'nov': '11', 'dec': '12',
        # Variants
        'sept': '9', 'januar': '1', 'janurary': '1',
        'july': '7',
        # Full names
        'january': '1', 'february': '2', 'march': '3', 'april': '4',
        'june': '6', 'august': '8', 'september': '9', 'october': '10',
        'november': '11', 'december': '12',
        # Non-English
        'abr': '4', 'mai': '5', 'avril': '4',
        # Seasons (map to approximate months)
        'winter': '1', 'spring': '4', 'summer': '7',
        'autumn': '10', 'fall': '10',
    }

    # Normalize to lowercase for lookup
    normalized = month_str.lower().strip()

    # Handle month ranges (e.g., "Jan-Feb") - take first month
    if '-' in normalized:
        normalized = normalized.split('-')[0].strip()

    result = month_map.get(normalized, '')
    return int(result) if result else ''  # Return empty string for unknown months


def parse_medline_file(input_file, output_csv):
    """
    Parse MEDLINE format file and extract article information
    """

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into individual records by PMID
    records = re.split(r'\n(?=PMID-)', content)

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pmid', 'title', 'abstract', 'authors', 'journal',
                         'year', 'doi', 'month', 'day', 'publication_types'])

        article_count = 0

        for record in records:
            if not record.strip():
                continue

            try:
                # Initialize fields
                pmid = ''
                title = ''
                abstract = ''
                authors = []
                publication_types = []
                journal = ''
                year = ''
                doi = ''
                month = ''
                day = ''

                # Extract PMID
                pmid_match = re.search(r'PMID-\s*(\d+)', record)
                if pmid_match:
                    pmid = pmid_match.group(1)

                # Extract Title (TI field)
                title_match = re.search(r'TI\s*-\s*(.+?)(?=\n[A-Z]{2,4}\s*-|\Z)', record, re.DOTALL)
                if title_match:
                    title = ' '.join(title_match.group(1).split())

                # Extract Abstract (AB field)
                abstract_match = re.search(r'AB\s*-\s*(.+?)(?=\n[A-Z]{2,4}\s*-|\Z)', record, re.DOTALL)
                if abstract_match:
                    abstract = ' '.join(abstract_match.group(1).split())

                # Extract Authors (AU fields)
                au_pattern = r'AU\s*-\s*(.+?)(?=\n[A-Z]{2,4}\s*-|\n\n|\Z)'
                au_matches = re.findall(au_pattern, record, re.DOTALL)
                for au in au_matches:
                    author_name = ' '.join(au.split())
                    if author_name:
                        authors.append(author_name)

                authors_str = '; '.join(authors)

                # Extract Publication Types (PT fields)
                pt_pattern = r'PT\s*-\s*(.+?)(?=\n[A-Z]{2,4}\s*-|\n\n|\Z)'
                pt_matches = re.findall(pt_pattern, record, re.DOTALL)
                for pt in pt_matches:
                    pub_type = ' '.join(pt.split())
                    if pub_type:
                        publication_types.append(pub_type)

                publication_types_str = '; '.join(publication_types)

                # Extract Journal (TA field)
                journal_match = re.search(r'TA\s*-\s*(.+?)(?=\n[A-Z]{2,4}\s*-|\Z)', record, re.DOTALL)
                if journal_match:
                    journal = ' '.join(journal_match.group(1).split())

                # Extract Date from DP field (e.g., "2016 Dec" or "2015 Jul")
                dp_match = re.search(r'DP\s*-\s*(\d{4})(?:\s+([A-Za-z-]+))?(?:\s+(\d+))?(?=\n)', record)
                if dp_match:
                    year = int(dp_match.group(1)) if dp_match.group(1) else ''
                    if dp_match.group(2):
                        month = convert_month_to_numeric(dp_match.group(2))
                    if dp_match.group(3):
                        day = int(dp_match.group(3)) if dp_match.group(3) else ''

                # Extract DOI from AID field
                doi_match = re.search(r'AID\s*-\s*([^\s]+)\s+\[doi\]', record)
                if doi_match:
                    doi = doi_match.group(1)

                # Write to CSV (year, month, day are integers when present, empty string when missing)
                writer.writerow([pmid, title, abstract, authors_str, journal,
                                 year, doi, month, day, publication_types_str])

                article_count += 1
                if article_count % 100 == 0:
                    print(f"Processed {article_count} articles...")

            except Exception as e:
                print(f"Error processing record: {e}")
                continue

        print(f"\nTotal articles processed: {article_count}")
        print(f"Output saved to: {output_csv}")


if __name__ == "__main__":
    input_file = "/Users/emirkarayagiz/Downloads/beyza_hoca_veri/beyza_hoca_pubmed.txt"  # Your MEDLINE format file
    output_csv = "beyza_hoca_veri.csv"

    print("Starting extraction...")
    parse_medline_file(input_file, output_csv)
    print("Done!")