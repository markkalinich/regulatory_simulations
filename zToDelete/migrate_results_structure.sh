#!/bin/bash
# Migration script to reorganize results folder structure
# Moves existing timestamped folders into new hierarchical structure

set -e

echo "=== Results Folder Migration ==="
echo "This will reorganize existing model comparison folders into:"
echo "  results/individual_prediction_performance/{experiment_type}/{timestamp}_{suffix}/"
echo ""

# Create new directory structure
mkdir -p results/individual_prediction_performance/suicide_ideation
mkdir -p results/individual_prediction_performance/therapy_request
mkdir -p results/individual_prediction_performance/therapy_engagement

echo "Created new directory structure"
echo ""

# Perform migrations using a simpler approach
migrate_folder() {
    local old_name="$1"
    local new_path="$2"
    local old_path="results/${old_name}"
    local new_full_path="results/individual_prediction_performance/${new_path}"
    
    if [ -d "$old_path" ]; then
        echo "Moving: $old_name"
        echo "  -> individual_prediction_performance/${new_path}"
        mv "$old_path" "$new_full_path"
    else
        echo "⚠️  Warning: $old_path not found, skipping"
    fi
}

# Perform all migrations
migrate_folder "20251023_therapy_engagement_fixed_v2" "therapy_engagement/20251023_therapy_engagement_fixed_v2_tx_engagement"
migrate_folder "20251024_192016_therapy_request_model_comparison" "therapy_request/20251024_192016_tx_request"
migrate_folder "20251024_203302_suicide_ideation_model_comparison" "suicide_ideation/20251024_203302_SI"
migrate_folder "20251024_204621_therapy_engagement_model_comparison" "therapy_engagement/20251024_204621_tx_engagement"
migrate_folder "20251024_204815_therapy_engagement_model_comparison" "therapy_engagement/20251024_204815_tx_engagement"
migrate_folder "20251024_204920_therapy_engagement_model_comparison" "therapy_engagement/20251024_204920_tx_engagement"
migrate_folder "20251025_230246_suicide_ideation_model_comparison" "suicide_ideation/20251025_230246_SI"
migrate_folder "20251025_230804_therapy_request_model_comparison" "therapy_request/20251025_230804_tx_request"
migrate_folder "20251025_232254_therapy_engagement_model_comparison" "therapy_engagement/20251025_232254_tx_engagement"

echo ""
echo "✅ Migration complete!"
echo ""
echo "New structure:"
tree -L 3 -d results/individual_prediction_performance 2>/dev/null || ls -R results/individual_prediction_performance

echo ""
echo "Folders that were NOT migrated (keeping as-is):"
ls -1d results/*/ 2>/dev/null | grep -v "individual_prediction_performance" | grep -v "zlegacy" | grep -v "experiments" | grep -v "visualizations" | grep -v "executive_summaries" || echo "  (none)"
