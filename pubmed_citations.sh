#!/bin/bash
# ============================================================
# FAST VERSION for macOS - Uses xargs for parallel execution
# No extra dependencies needed
# ============================================================

INPUT_FILE="misc/radiology_results2.txt"

OUTPUT_FILE="citations_output.csv"
TEMP_DIR="temp_elink_$$"
JOBS=3

# ============================================================

mkdir -p "$TEMP_DIR"

# Extract PMIDs (macOS compatible)
PMIDS=$(grep "^PMID-" "$INPUT_FILE" | sed 's/PMID-[[:space:]]*//' | sort -u)
COUNT=$(echo "$PMIDS" | grep -c '[0-9]')

echo "Found $COUNT PMIDs"
echo "Processing with $JOBS parallel jobs..."
echo ""

# Save PMIDs to temp file
echo "$PMIDS" > "$TEMP_DIR/pmids.txt"

# Process function (written to separate script for xargs)
cat > "$TEMP_DIR/process.sh" << 'SCRIPT'
#!/bin/bash
PMID=$1
OUTDIR=$2

CITED=$(elink -db pubmed -id "$PMID" -cited 2>/dev/null | xtract -pattern Id -element Id | tr '\t' ';')
CITES=$(elink -db pubmed -id "$PMID" -cites 2>/dev/null | xtract -pattern Id -element Id | tr '\t' ';')

echo "$PMID|$CITED|$CITES" > "$OUTDIR/$PMID.txt"
echo "Done: $PMID"
SCRIPT
chmod +x "$TEMP_DIR/process.sh"

# Run in parallel using xargs
cat "$TEMP_DIR/pmids.txt" | xargs -P $JOBS -I {} "$TEMP_DIR/process.sh" {} "$TEMP_DIR"

# Combine results
echo ""
echo "Combining results..."
echo "pmid,cited_by_count,cited_by_pmids,cites_count,cites_pmids" > "$OUTPUT_FILE"

for PMID in $PMIDS; do
    if [ -f "$TEMP_DIR/$PMID.txt" ]; then
        LINE=$(cat "$TEMP_DIR/$PMID.txt")
        CITED=$(echo "$LINE" | cut -d'|' -f2)
        CITES=$(echo "$LINE" | cut -d'|' -f3)

        if [ -z "$CITED" ]; then
            CITED_COUNT=0
        else
            CITED_COUNT=$(echo "$CITED" | tr ';' '\n' | grep -c '[0-9]')
        fi

        if [ -z "$CITES" ]; then
            CITES_COUNT=0
        else
            CITES_COUNT=$(echo "$CITES" | tr ';' '\n' | grep -c '[0-9]')
        fi

        echo "$PMID,$CITED_COUNT,$CITED,$CITES_COUNT,$CITES" >> "$OUTPUT_FILE"
    fi
done

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "Done! Results saved to $OUTPUT_FILE"