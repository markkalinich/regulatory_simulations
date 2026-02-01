# Regulatory Paper Cache v3

This database contains the cached results for 14 models used in the regulatory simulation paper.

## Database Information

- **File:** `results.db`
- **Size:** 37 MB
- **Models:** 14 (all Q8_0 quantization)
- **Cache Entries:** 23,100 total
- **MD5 Checksum:** See `MD5SUM.txt`

## Models Included

### Gemma (5 models)
- google.gemma-3-270m-it
- google.gemma-3-1b-it
- google.gemma-3-4b-it
- google.gemma-3-12b-it
- google.gemma-3-27b-it

### Qwen (6 models)
- qwen3-0.6b
- unsloth/qwen3-1.7b (Q8_0)
- unsloth/qwen3-4b (Q8_0)
- qwen/qwen3-8b-gguf/qwen3-8b-q8_0.gguf
- qwen/qwen3-14b
- qwen/qwen3-32b

### Llama (3 models)
- bartowski/llama-3.2-1b-instruct
- meta-llama-3.1-8b-instruct
- meta/llama-3.3-70b

## Experiments

- **SI (Suicidal Ideation):** 450 inputs × 14 models = 6,300 entries
- **TR (Therapy Request):** 780 inputs × 14 models = 10,920 entries
- **TE (Therapy Engagement):** 420 inputs × 14 models = 5,880 entries

## Success Rates

- Successful: 22,251 (96.32%)
- Parse failures: 849 (3.68%)

## Verification

To verify database integrity:

```bash
md5sum -c MD5SUM.txt
```

Expected output: `results.db: OK`

## Created

January 19-20, 2026
