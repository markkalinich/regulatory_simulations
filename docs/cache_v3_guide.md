# Cache v3 System

## Overview

Cache v3 uses LM Studio's `path` field as the model identifier, which includes publisher, model name, quantization, and filename (e.g., `DevQuasar/google.gemma-3-270m-it-GGUF/google.gemma-3-270m-it.Q8_0.gguf`).

## Cache Key Components

| Field | Source |
|-------|--------|
| `model_path` | LM Studio `path` field |
| `quantization_name` | LM Studio quantization (e.g., Q8_0) |
| `prompt_hash` | MD5 of prompt content |
| `input_hash` | MD5 of input text |
| `temperature` | API parameter |
| `max_tokens` | API parameter |
| `top_p` | API parameter |
| `context_length` | API parameter |

Metadata stored but not part of cache key: `hostname`, `prompt_suffix_applied`, `execution_context`.

## Database Location

`regulatory_paper_cache_v3/results.db`

## Usage

### Check cache status
```bash
python -m utilities.batch_cache_checker_v2 \
    --prompt-name "system_suicide_detection_v2" \
    --prompt-file "data/prompts/system_suicide_detection_v2.txt" \
    --input-data "data/inputs/finalized_input_data/SI_finalized_sentences.csv" \
    --cache-dir "regulatory_paper_cache_v3"
```

### Query cache directly
```python
import sqlite3
conn = sqlite3.connect('regulatory_paper_cache_v3/results.db')
cursor = conn.cursor()

cursor.execute('''
    SELECT ck.model_path, cr.status_type, COUNT(*)
    FROM cached_results cr
    JOIN cache_keys ck ON cr.cache_id = ck.cache_id
    GROUP BY ck.model_path, cr.status_type
''')
for row in cursor.fetchall():
    print(row)
```

## Files

| File | Purpose |
|------|---------|
| `cache/result_cache_v2.py` | Cache implementation |
| `utilities/cache_checker_v2.py` | Single model check |
| `utilities/batch_cache_checker_v2.py` | Batch check |

## Common Issues

**"Model not found"**: Ensure LM Studio is running and model is downloaded.

**Cache misses for same model**: Check if `path` changed (`lms ls --json`).
