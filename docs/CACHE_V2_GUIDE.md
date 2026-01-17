# Cache V2 System

## Background

Cache V1 used `model_family + model_size + model_version` as cache keys. This caused reproducibility issues:
- Different quantizations (Q4 vs Q8) shared cache entries incorrectly
- `model_full_name` was unstable across runs
- Cache misses occurred for logically identical experiments

Cache V2 uses LM Studio's `path` field as the model identifier. This path includes publisher, model name, quantization, and filename (e.g., `DevQuasar/google.gemma-3-270m-it-GGUF/google.gemma-3-270m-it.Q8_0.gguf`).

## Cache Key Components

| Field | Source |
|-------|--------|
| `model_path` | LM Studio `path` field |
| `prompt_hash` | MD5 of prompt content (before model-specific modifications) |
| `input_hash` | MD5 of input text |
| `temperature` | API parameter |
| `max_tokens` | API parameter |
| `top_p` | API parameter |
| `context_length` | API parameter |

Metadata stored but NOT part of cache key: `hostname`, `prompt_suffix_applied`, `execution_context`.

## Database Location

- V1: `cache/results.db`
- V2: `cache_v2/results.db`

Both can coexist. No automatic migration.

## Usage

### Run experiments
```bash
./bash_scripts/run_experiments.sh \
    --models "gemma:270m-it,qwen:0.6b" \
    data/inputs/finalized_input_data/SI_finalized_sentences.csv \
    data/prompts/system_suicide_detection_v2.txt \
    system_suicide_detection_v2
```

### Check cache status
```bash
python -m utilities.batch_cache_checker_v2 \
    --prompt-name "system_suicide_detection_v2" \
    --prompt-file "data/prompts/system_suicide_detection_v2.txt" \
    --input-data "data/inputs/finalized_input_data/SI_finalized_sentences.csv" \
    --cache-dir "cache_v2"
```

### Query cache directly
```python
import sqlite3
conn = sqlite3.connect('cache_v2/results.db')
cursor = conn.cursor()

# Count by model and status
cursor.execute('''
    SELECT ck.model_path, cr.status_type, COUNT(*)
    FROM cached_results cr
    JOIN cache_keys ck ON cr.cache_id = ck.cache_id
    GROUP BY ck.model_path, cr.status_type
''')
for row in cursor.fetchall():
    print(row)
```

### Delete cache entries
```python
import sqlite3
conn = sqlite3.connect('cache_v2/results.db')
cursor = conn.cursor()

cursor.execute('''
    DELETE FROM cached_results 
    WHERE cache_id IN (
        SELECT cache_id FROM cache_keys 
        WHERE model_path LIKE '%gemma-3-270m-it%'
    )
''')
cursor.execute('''
    DELETE FROM cache_keys WHERE model_path LIKE '%gemma-3-270m-it%'
''')
conn.commit()
```

## Design Notes

1. **Prompt hashing**: Computed before model-specific modifications (e.g., Qwen `/no_think` suffix). The suffix is stored in `prompt_suffix_applied`.

2. **API errors**: Counted as cache entries for reproducibility. Preflight reports them separately.

3. **Model path**: Relative path from LM Studio models directory, consistent across machines with same model files.

## Files

| File | Purpose |
|------|---------|
| `cache/result_cache_v2.py` | Cache implementation |
| `utilities/cache_checker_v2.py` | Single model check |
| `utilities/batch_cache_checker_v2.py` | Batch check |

## Common Issues

**"Model not found"**: Ensure LM Studio is running and model is downloaded.

**Cache misses for same model**: Check if `path` changed (`lms ls --json`).

**Wrong database**: V2 uses `results.db`, not `cache.db`.

## Implementation Notes

Issues encountered during development:

1. Bash `((count++))` fails with `set -e` when count=0. Use `count=$((count + 1))`.

2. Schema uses `cache_id` as foreign key, not `cache_key_id`.

3. Python logging outputs to stderr. Captured by `2>&1` in bash.

4. LM Studio `modelKey` ≠ `path`. Cache uses `path`.
