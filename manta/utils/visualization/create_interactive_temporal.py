#!/usr/bin/env python3
"""
Generate a standalone HTML file with embedded CSV and MD data.
This creates a single HTML file that can be shared and opened without a server.
"""

import argparse
import csv
import json
import re
from pathlib import Path


def read_csv_to_json(csv_path):
    """Read CSV file and convert to JSON array."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = []
        for row in reader:
            # Convert numeric values
            processed_row = {}
            for key, value in row.items():
                if key.lower() == "period":
                    processed_row[key] = value
                else:
                    try:
                        processed_row[key] = int(value) if value else 0
                    except ValueError:
                        processed_row[key] = value
            data.append(processed_row)
        return data


def parse_topic_names_from_md(md_path):
    """Parse topic names from markdown file."""
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Pattern to match: - **Konu X:** Description
    pattern = r"-\s*\*\*Konu\s+(\d+):\*\*\s*(.+?)(?=\n\s*\n|-\s*\*\*Konu|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)

    topic_names = {}
    for match in matches:
        topic_num = match[0].strip()
        description = match[1].strip()

        # Create mapping from "Topic X" to full description
        csv_key = f"Topic {topic_num}"
        display_name = f"Konu {topic_num}: {description}"

        topic_names[csv_key] = display_name

    return topic_names


def read_markdown_content(md_path):
    """Read the full markdown content."""
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()


def escape_for_js_string(text):
    """Escape text for use in JavaScript template literal."""
    # Escape backticks and backslashes
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`")
    text = text.replace("$", "\\$")
    return text


def generate_temporal_line_graph(CSV_PATH, OUTPUT_HTML):
    """Generate the standalone HTML file."""

    # Determine CSV path
    csv_path = Path(CSV_PATH)
    script_dir = Path(__file__).parent.resolve()

    # Read all data
    print(f"Reading CSV data from {csv_path}...")
    csv_data = read_csv_to_json(csv_path)

    print("Parsing topic names...")
    topic_names = parse_topic_names_from_md(script_dir / "md" / "report.md")

    print("Reading markdown content...")
    markdown_content = read_markdown_content(script_dir / "md" / "report.md")

    # Read the current index.html
    print("Reading index.html template...")
    with open(script_dir / "index.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    # Create the embedded data section
    embedded_data_js = f"""
    // ===== EMBEDDED DATA =====
    // This data is embedded for standalone operation
    // Generated from csv/data.csv and md/report.md

    const EMBEDDED_CSV_DATA = {json.dumps(csv_data, ensure_ascii=False, indent=2)};

    const EMBEDDED_TOPIC_NAMES = {json.dumps(topic_names, ensure_ascii=False, indent=2)};

    const EMBEDDED_MARKDOWN = `{escape_for_js_string(markdown_content)}`;

    // ===== END EMBEDDED DATA =====
    """

    # Find the script tag and insert embedded data
    script_start = html_content.find("<script>")
    if script_start == -1:
        print("Error: Could not find <script> tag")
        return

    # Insert embedded data right after <script> tag
    insert_pos = script_start + len("<script>")
    new_html = html_content[:insert_pos] + embedded_data_js + html_content[insert_pos:]

    # Replace loadTopicNames function completely
    old_load_topic_names = """        // Load topic names from API
        async function loadTopicNames() {
            try {
                const response = await fetch('/api/topic-names');
                if (!response.ok) {
                    throw new Error('Topic names API isteği başarısız oldu');
                }
                const result = await response.json();

                console.log('Topic Names Response:', result);

                if (!result.success) {
                    throw new Error(result.error || 'Bilinmeyen hata');
                }

                topicNames = result.topics;
                console.log('Topic Names loaded:', topicNames);
            } catch (error) {
                console.error('Topic Names Error:', error);
                // Continue even if topic names fail to load
            }
        }"""

    new_load_topic_names = """        // Load topic names from embedded data
        async function loadTopicNames() {
            try {
                // Using embedded data instead of API
                const result = { success: true, topics: EMBEDDED_TOPIC_NAMES };

                console.log('Topic Names Response:', result);

                if (!result.success) {
                    throw new Error(result.error || 'Bilinmeyen hata');
                }

                topicNames = result.topics;
                console.log('Topic Names loaded:', topicNames);
            } catch (error) {
                console.error('Topic Names Error:', error);
                // Continue even if topic names fail to load
            }
        }"""

    new_html = new_html.replace(old_load_topic_names, new_load_topic_names)

    # Replace loadCSV function completely
    old_load_csv = """        // Load and parse CSV from API
        async function loadCSV() {
            try {
                const response = await fetch('/api/data');
                if (!response.ok) {
                    throw new Error('API isteği başarısız oldu');
                }
                const result = await response.json();

                console.log('API Response:', result);

                if (!result.success) {
                    throw new Error(result.error || 'Bilinmeyen hata');
                }

                csvData = result.data;
                console.log('CSV Data:', csvData);
                processCSVData();
            } catch (error) {
                console.error('CSV Error:', error);
                showError('CSV yükleme hatası: ' + error.message);
            }
        }"""

    new_load_csv = """        // Load and parse CSV from embedded data
        async function loadCSV() {
            try {
                // Using embedded data instead of API
                const result = { success: true, data: EMBEDDED_CSV_DATA };

                console.log('API Response:', result);

                if (!result.success) {
                    throw new Error(result.error || 'Bilinmeyen hata');
                }

                csvData = result.data;
                console.log('CSV Data:', csvData);
                processCSVData();
            } catch (error) {
                console.error('CSV Error:', error);
                showError('CSV yükleme hatası: ' + error.message);
            }
        }"""

    new_html = new_html.replace(old_load_csv, new_load_csv)

    # Replace loadMarkdown function completely
    old_load_markdown = """        // Load markdown report
        async function loadMarkdown() {
            try {
                const response = await fetch('md/report.md');
                if (!response.ok) {
                    throw new Error('Markdown dosyası bulunamadı');
                }
                const markdown = await response.text();

                // Use marked() function directly - compatible with multiple versions
                const html = typeof marked.parse === 'function'
                    ? marked.parse(markdown)
                    : marked(markdown);

                document.getElementById('reportContent').innerHTML = html;
            } catch (error) {
                console.error('Markdown Error:', error);
                document.getElementById('reportContent').innerHTML =
                    '<div class="error">Rapor yüklenemedi: ' + error.message + '</div>';
            }
        }"""

    new_load_markdown = """        // Load markdown report from embedded data
        async function loadMarkdown() {
            try {
                // Using embedded data instead of API
                const markdown = EMBEDDED_MARKDOWN;

                // Use marked() function directly - compatible with multiple versions
                const html = typeof marked.parse === 'function'
                    ? marked.parse(markdown)
                    : marked(markdown);

                document.getElementById('reportContent').innerHTML = html;
            } catch (error) {
                console.error('Markdown Error:', error);
                document.getElementById('reportContent').innerHTML =
                    '<div class="error">Rapor yüklenemedi: ' + error.message + '</div>';
            }
        }"""

    new_html = new_html.replace(old_load_markdown, new_load_markdown)

    # Write standalone HTML
    output_path = script_dir / "index_standalone.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_html)

    print(f"\n✓ Standalone HTML created: {output_path}")
    print(f"✓ File size: {len(new_html) / 1024:.1f} KB")
    print(f"✓ CSV rows: {len(csv_data)}")
    print(f"✓ Topics: {len(topic_names)}")
    print("\nYou can now share index_standalone.html - it works without a server!")


if __name__ == "__main__":
    CSV = ""
    OUTPUT = ""
    generate_temporal_line_graph(CSV, OUTPUT)
