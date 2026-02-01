#!/bin/bash
#
# Experiment Runner for Safety Simulation Experiments
#
# This script runs the actual experiments. It expects preflight validation
# to have already passed (either by sourcing preflight.sh or running it first).
#
# Usage:
#   # Option 1: Source preflight first (recommended)
#   source preflight.sh [--models "family:size,..."] INPUT_DATA PROMPT_FILE PROMPT_NAME
#   ./run_experiments.sh
#
#   # Option 2: Pass arguments directly (runs own validation)
#   ./run_experiments.sh [--models "family:size,..."] INPUT_DATA PROMPT_FILE PROMPT_NAME
#
# Required environment variables (set by preflight.sh):
#   INPUT_DATA, PROMPT_FILE, PROMPT_NAME, EXPERIMENT_TYPE, SUFFIX, TIMESTAMP

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set PROJECT_ROOT if not already set (from preflight.sh)
if [[ -z "${PROJECT_ROOT}" ]]; then
    export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

# Helper function to run Python modules from project root
run_python_module() {
    (cd "${PROJECT_ROOT}" && python -m "$@")
}

# =============================================================================
# Check if preflight was run or we need to parse arguments
# =============================================================================

if [[ -z "${INPUT_DATA}" ]] || [[ -z "${PROMPT_FILE}" ]] || [[ -z "${PROMPT_NAME}" ]]; then
    # Preflight wasn't sourced - check if arguments were passed
    if [ $# -ge 3 ]; then
        echo "ℹ️  Running preflight validation..."
        source "${SCRIPT_DIR}/preflight.sh"
        run_preflight "$@" || {
            echo "❌ Preflight validation failed. Cannot run experiments."
            exit 1
        }
    else
        echo "❌ Error: Required environment variables not set and no arguments provided."
        echo ""
        echo "Either source preflight.sh first:"
        echo "  source preflight.sh [--models \"family:size,...\"] INPUT_DATA PROMPT_FILE PROMPT_NAME"
        echo "  ./run_experiments.sh"
        echo ""
        echo "Or pass arguments directly:"
        echo "  ./run_experiments.sh [--models \"family:size,...\"] INPUT_DATA PROMPT_FILE PROMPT_NAME"
        exit 1
    fi
fi

# Set defaults if not already set
NUM_REPLICATES="${NUM_REPLICATES:-1}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

# Convert relative paths to absolute paths (Python runs from PROJECT_ROOT)
if [[ ! "$INPUT_DATA" = /* ]]; then
    INPUT_DATA="$(cd "$(dirname "$INPUT_DATA")" && pwd)/$(basename "$INPUT_DATA")"
fi
if [[ ! "$PROMPT_FILE" = /* ]]; then
    PROMPT_FILE="$(cd "$(dirname "$PROMPT_FILE")" && pwd)/$(basename "$PROMPT_FILE")"
fi

# Track experiment results
declare -a FAILED_MODELS=()
declare -a SKIPPED_MODELS=()
declare -a SUCCESSFUL_MODELS=()

# =============================================================================
# Output Directory Setup
# =============================================================================

OUTPUT_DIR="results/individual_prediction_performance/${EXPERIMENT_TYPE}/${TIMESTAMP}_${SUFFIX}"
echo "Creating output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"/{plots,tables,confusion_matrices,reports}

# Log file - redirect all output
LOG_FILE="${OUTPUT_DIR}/analysis.log"
exec > >(tee -a "${LOG_FILE}")
exec 2>&1

echo "═══════════════════════════════════════════════════════════════════"
echo "  EXPERIMENT RUN STARTED"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Timestamp:    $(date)"
echo "Experiment:   ${EXPERIMENT_TYPE}"
echo "Input data:   ${INPUT_DATA}"
echo "Prompt:       ${PROMPT_FILE}"
echo "Output:       ${OUTPUT_DIR}"
echo ""

# =============================================================================
# Model Registry Functions (thin wrappers)
# =============================================================================

lookup_model_spec() {
    run_python_module config.models_registry --family "$1" --size "$2" --get-spec 2>/dev/null
}

get_all_families() {
    run_python_module config.models_registry --list-families 2>/dev/null
}

get_family_sizes() {
    run_python_module config.models_registry --family "$1" --list-sizes 2>/dev/null
}

# =============================================================================
# LM Studio Management
# =============================================================================

unload_all_models() {
    lms unload --all 2>/dev/null || true
}

ensure_model_loaded() {
    # Use Python wrapper for loading (handles validation and full path resolution)
    run_python_module utilities.lms_manager --load "$1"
}

unload_model() {
    local model_name=$1
    lms unload "${model_name}" 2>/dev/null || true
}

# =============================================================================
# Cache Checking (V2 - path-based)
# =============================================================================

check_experiment_cache_v3() {
    local model_key=$1 family=$2
    run_python_module utilities.cache_checker_v2 \
        --model-key "${model_key}" \
        --model-family "${family}" \
        --prompt-file "${PROMPT_FILE}" \
        --input-data "${INPUT_DATA}" \
        --num-replicates "${NUM_REPLICATES}" \
        --cache-dir "regulatory_paper_cache_v3" \
        --quiet 2>/dev/null || echo "0.0"
}

# =============================================================================
# Run Single Model Experiment
# =============================================================================

run_model_experiment() {
    local family=$1
    local size=$2
    local current_num=${3:-}
    local total_num=${4:-}
    local model_key="${family}:${size}"

    local spec
    spec=$(lookup_model_spec "${family}" "${size}") || {
        echo "  ✗ Missing registry entry for ${model_key}; skipping"
        SKIPPED_MODELS+=("${model_key} (not in registry)")
        return 0
    }
    
    local version="${spec%%|*}"
    local model_name="${spec#*|}"

    echo ""
    echo "─────────────────────────────────────────────────────────────────"
    if [ -n "${current_num}" ] && [ -n "${total_num}" ]; then
        echo "  Processing: ${model_key} (${current_num}/${total_num})"
    else
        echo "  Processing: ${model_key}"
    fi
    echo "─────────────────────────────────────────────────────────────────"
    echo "  Version:    ${version}"
    echo "  Model ID:   ${model_name}"

    # Check V2 cache status
    local cache_percentage
    cache_percentage=$(check_experiment_cache_v3 "${model_name}" "${family}" | tail -1)
    
    # Validate cache percentage
    local cache_int=0
    if [[ "${cache_percentage}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        cache_int="${cache_percentage%.*}"
    else
        echo "  ⚠️  Cache check failed, assuming 0%"
        cache_percentage="0.0"
    fi
    
    # Load model if needed
    local model_loaded=false
    if [ "${cache_int}" -eq 100 ]; then
        echo "  ✓ Cache: 100% - skipping model load"
    else
        echo "  → Cache: ${cache_percentage}% - loading model..."
        if ! ensure_model_loaded "${model_name}"; then
            echo "  ✗ Failed to load model - skipping"
            SKIPPED_MODELS+=("${model_key} (failed to load)")
            return 0
        fi
        model_loaded=true
        
        # Validate quantization matches expected
        echo "  → Validating quantization..."
        if ! run_python_module utilities.validate_quantization "${family}" "${size}" "${model_name}"; then
            echo "  ✗ Quantization validation failed - skipping"
            SKIPPED_MODELS+=("${model_key} (wrong quantization)")
            # Unload the incorrectly loaded model
            unload_model "${model_name}"
            return 0
        fi
    fi

    # Run experiment with V2 cache
    local exit_code=0
    run_python_module orchestration.run_experiment \
        --experiment-name "${family}_${size}_${PROMPT_NAME}_analysis" \
        --model-family "${family}" \
        --model-size "${size}" \
        --model-version "${version}" \
        --prompt-name "${PROMPT_NAME}" \
        --input "${INPUT_DATA}" \
        --system "${PROMPT_FILE}" \
        --num-replicates ${NUM_REPLICATES} \
        --model-key "${model_name}" \
        --cache-dir "regulatory_paper_cache_v3" \
        --description "Analysis: ${family} ${size}" || exit_code=$?
    
    # Unload model if we loaded it
    if [ "$model_loaded" = true ]; then
        echo "  → Unloading model..."
        unload_model "${model_name}"
        sleep 1
    fi
    
    if [ $exit_code -eq 0 ]; then
        echo "  ✓ Completed successfully"
        SUCCESSFUL_MODELS+=("${model_key}")
    else
        echo "  ✗ Experiment failed (exit code: $exit_code)"
        FAILED_MODELS+=("${model_key} (exit code: $exit_code)")
    fi
}

# =============================================================================
# Main Experiment Loop
# =============================================================================

run_experiments() {
    echo ""
    echo "Starting experiments..."
    
    # Unload any existing models
    unload_all_models
    
    if [ -n "${MODELS_OVERRIDE}" ]; then
        # Run specific models
        echo ""
        echo "📋 Running specified models: ${MODELS_OVERRIDE}"
        
        IFS=',' read -ra MODEL_ARRAY <<< "${MODELS_OVERRIDE}"
        local total_models=${#MODEL_ARRAY[@]}
        local current_model=0
        
        for model_pair in "${MODEL_ARRAY[@]}"; do
            current_model=$((current_model + 1))
            local family="${model_pair%%:*}"
            local size="${model_pair#*:}"
            run_model_experiment "${family}" "${size}" "${current_model}" "${total_models}"
            sleep 2
        done
        
        unload_all_models
    else
        # Run all enabled models
        echo ""
        echo "📋 Running all enabled models from registry"
        
        local all_families
        all_families=$(get_all_families)
        
        for family in ${all_families}; do
            local family_upper
            family_upper=$(printf '%s' "${family}" | tr '[:lower:]' '[:upper:]')
            
            echo ""
            echo "═══════════════════════════════════════════════════════════════════"
            echo "  FAMILY: ${family_upper}"
            echo "═══════════════════════════════════════════════════════════════════"
            
            local sizes_list
            sizes_list=$(get_family_sizes "${family}") || {
                echo "  ⚠️  No sizes configured; skipping"
                continue
            }
            
            for size in ${sizes_list}; do
                run_model_experiment "${family}" "${size}"
                sleep 2
            done
            
            # Unload after each family to free memory
            unload_all_models
            
            echo ""
            echo "✓ ${family_upper} family complete"
        done
    fi
}

# =============================================================================
# Generate Analysis
# =============================================================================

generate_analysis() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  GENERATING ANALYSIS"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    
    # Build analysis command args
    local analysis_args=(
        analysis.model_performance.batch_results_analyzer
        --output-dir "${OUTPUT_DIR}"
        --input-data "${INPUT_DATA}"
        --prompt-file "${PROMPT_FILE}"
        --timestamp "${TIMESTAMP}"
        --experiment-type "${EXPERIMENT_TYPE}"
        --results-timestamp "${TIMESTAMP}"
        --cache-dir "regulatory_paper_cache_v3"
    )
    
    # Pass models filter if specific models were requested
    if [ -n "${MODELS_OVERRIDE}" ]; then
        analysis_args+=(--models "${MODELS_OVERRIDE}")
    fi
    
    run_python_module "${analysis_args[@]}"
}

# =============================================================================
# Summary Report
# =============================================================================

print_summary() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  EXPERIMENT RUN COMPLETE"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    echo "  Results: ${OUTPUT_DIR}"
    echo "  Log:     ${LOG_FILE}"
    echo ""
    echo "  ✓ Successful: ${#SUCCESSFUL_MODELS[@]}"
    
    if [ ${#SKIPPED_MODELS[@]} -gt 0 ]; then
        echo "  ⚠ Skipped:    ${#SKIPPED_MODELS[@]}"
        for model in "${SKIPPED_MODELS[@]}"; do
            echo "      - ${model}"
        done
    fi
    
    if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
        echo "  ✗ Failed:     ${#FAILED_MODELS[@]}"
        for model in "${FAILED_MODELS[@]}"; do
            echo "      - ${model}"
        done
    fi
    echo ""
}

# =============================================================================
# Main Entry Point
# =============================================================================

main() {
    run_experiments
    
    # Only run analysis if at least one model succeeded
    if [ ${#SUCCESSFUL_MODELS[@]} -gt 0 ]; then
        generate_analysis
    else
        echo ""
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  ⚠️  SKIPPING ANALYSIS - No successful experiments to analyze"
        echo "═══════════════════════════════════════════════════════════════════"
        echo ""
    fi
    
    print_summary
    
    # Return non-zero if any failures
    if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
        return 1
    fi
}

main
