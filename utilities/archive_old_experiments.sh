#!/bin/bash
# Archive old experiment files that don't use finalized datasets
# Keeps only experiments using current validated data in main experiments/ folder

set -e

echo "=== Archiving Old Experiments ==="
echo "This will move non-finalized experiments to data/model_outputs/archive/"
echo ""

# Create archive directory
ARCHIVE_DIR="data/model_outputs/archive"
mkdir -p "$ARCHIVE_DIR"

# Define patterns for CURRENT finalized datasets we want to KEEP in main folder
KEEP_PATTERNS=(
    "*SI_finalized_sentences*"
    "*therapy_request_finalized_sentences*"
    "*therapy_engagement_conversations_balanced_145*"
)

echo "Keeping experiments that match finalized datasets:"
echo "  - SI_finalized_sentences (450 sentences)"
echo "  - therapy_request_finalized_sentences (780 sentences)"
echo "  - therapy_engagement_conversations_balanced_145 (435 conversations)"
echo ""

# Count files before
TOTAL_FILES=$(ls data/model_outputs/*.csv 2>/dev/null | wc -l | tr -d ' ')
echo "Total experiment files (CSV): $TOTAL_FILES"

# Find files to keep
KEEP_COUNT=0
for pattern in "${KEEP_PATTERNS[@]}"; do
    count=$(ls data/model_outputs/$pattern 2>/dev/null | wc -l | tr -d ' ')
    KEEP_COUNT=$((KEEP_COUNT + count))
done

ARCHIVE_COUNT=$((TOTAL_FILES - KEEP_COUNT))
echo "Files to keep: $KEEP_COUNT"
echo "Files to archive: $ARCHIVE_COUNT"
echo ""

read -p "Proceed with archiving? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Archive files NOT matching keep patterns
archived=0
for csv_file in data/model_outputs/*.csv; do
    # Get base filename without path
    base_name=$(basename "$csv_file")
    
    # Check if file matches any keep pattern
    should_keep=false
    for pattern in "${KEEP_PATTERNS[@]}"; do
        if [[ "$base_name" == ${pattern} ]]; then
            should_keep=true
            break
        fi
    done
    
    # If not keeping, archive all 3 files (CSV, JSONL, JSON)
    if [[ "$should_keep" == false ]]; then
        # Extract experiment ID (remove _results.csv)
        exp_id="${base_name%_results.csv}"
        
        # Move all three files
        if [ -f "data/model_outputs/${exp_id}_results.csv" ]; then
            mv "data/model_outputs/${exp_id}_results.csv" "$ARCHIVE_DIR/"
        fi
        if [ -f "data/model_outputs/${exp_id}_results.jsonl" ]; then
            mv "data/model_outputs/${exp_id}_results.jsonl" "$ARCHIVE_DIR/"
        fi
        if [ -f "data/model_outputs/${exp_id}_summary.json" ]; then
            mv "data/model_outputs/${exp_id}_summary.json" "$ARCHIVE_DIR/"
        fi
        
        archived=$((archived + 1))
        if [ $((archived % 10)) -eq 0 ]; then
            echo "  Archived $archived experiments..."
        fi
    fi
done

echo ""
echo "✅ Archive complete!"
echo ""
echo "Summary:"
echo "  Archived: $archived experiment sets ($(($archived * 3)) files)"
echo "  Remaining in experiments/: $KEEP_COUNT experiment sets"
echo ""
echo "Verification:"
ls data/model_outputs/*.csv 2>/dev/null | wc -l | xargs echo "  Active CSV files:"
ls "$ARCHIVE_DIR"/*.csv 2>/dev/null | wc -l | xargs echo "  Archived CSV files:"
echo ""
echo "Archive location: $ARCHIVE_DIR"
