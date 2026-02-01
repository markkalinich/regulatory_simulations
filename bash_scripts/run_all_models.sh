#!/bin/bash
#
# Complete Multi-Model Safety Simulation Pipeline
#
# This is the main entry point for running safety simulation experiments.
# It combines preflight validation and experiment execution.
#
# Usage:
#   ./run_all_models.sh [OPTIONS] INPUT_DATA PROMPT_FILE PROMPT_NAME
#
# Options:
#   --preflight-only    Only run validation, don't execute experiments
#   --models MODELS     Comma-separated list of family:size pairs to run
#   --replicates N      Number of replicates per sample (default: 1)
#
# For separate control, use:
#   ./preflight.sh       - Validation only
#   ./run_experiments.sh - Execution only (after preflight)
#
# Examples:
#   # Run all enabled models on Suicidal Ideation
#   ./run_all_models.sh \
#       data/inputs/finalized_input_data/SI_finalized_sentences.csv \
#       data/prompts/system_suicide_detection_v2.txt \
#       system_suicide_detection_v2
#
#   # Run specific models only
#   ./run_all_models.sh --models "gemma:270m-it,gemma:1b-it,qwen:0.6b" \
#       data/inputs/finalized_input_data/SI_finalized_sentences.csv \
#       data/prompts/system_suicide_detection_v2.txt \
#       system_suicide_detection_v2
#
#   # Just validate (don't run)
#   ./run_all_models.sh --preflight-only \
#       data/inputs/finalized_input_data/SI_finalized_sentences.csv \
#       data/prompts/system_suicide_detection_v2.txt \
#       system_suicide_detection_v2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Parse Options
# =============================================================================

PREFLIGHT_ONLY=false
EXTRA_ARGS=""

while [[ "$1" == --* ]]; do
    case "$1" in
        --preflight-only)
            PREFLIGHT_ONLY=true
            shift
            ;;
        --models)
            EXTRA_ARGS="${EXTRA_ARGS} --models $2"
            shift 2
            ;;
        --replicates)
            EXTRA_ARGS="${EXTRA_ARGS} --replicates $2"
            shift 2
            ;;
        --help|-h)
            head -50 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Main
# =============================================================================

# Run preflight validation
echo ""
source "${SCRIPT_DIR}/preflight.sh"
# Pass options (--models, --replicates) then positional args
run_preflight ${EXTRA_ARGS} "$@" || {
    echo ""
    echo "❌ Preflight validation failed. Aborting."
    exit 1
}

# Exit here if preflight-only mode
if [ "$PREFLIGHT_ONLY" = true ]; then
    echo ""
    echo "ℹ️  Preflight-only mode: not running experiments."
    echo "   To run experiments, omit --preflight-only"
    exit 0
fi

# Run experiments
echo ""
source "${SCRIPT_DIR}/run_experiments.sh"
