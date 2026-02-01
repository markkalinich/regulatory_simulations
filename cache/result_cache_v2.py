#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result Cache V2 - Path-based model identification

Key changes from v1:
1. Uses LM Studio's `path` as the unique model identifier (not family/size/version)
2. Pulls model metadata from LM Studio API at cache time
3. Stores prompt modifications (like /no_think) as metadata
4. Tracks execution context (hostname, etc.) without affecting cache key

Cache key is based on:
- model_path (from LM Studio)
- quantization_name (e.g., Q8_0, Q4_K_M) - prevents collision for multi-quant folders
- prompt_hash (content BEFORE model-specific modifications)
- input_hash
- temperature, max_tokens, top_p, context_length

Note: Adding quantization_name to the cache key (Jan 2026) means existing cache entries
will NOT match new cache lookups. New results will be stored with new cache IDs.
"""

import hashlib
import json
import sqlite3
import subprocess
import socket
import platform
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging


def get_lm_studio_models() -> Dict[str, Dict[str, Any]]:
    """Fetch model metadata from LM Studio CLI."""
    try:
        result = subprocess.run(['lms', 'ls', '--json'], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logging.warning(f"LM Studio CLI error: {result.stderr}")
            return {}
        models = json.loads(result.stdout)
        # Index by modelKey for lookup
        return {m['modelKey']: m for m in models}
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
        logging.warning(f"Could not fetch LM Studio models: {e}")
        return {}


def get_execution_context() -> Dict[str, Any]:
    """Get current execution environment info."""
    return {
        'hostname': socket.gethostname(),
        'os_type': platform.system().lower(),
        'os_version': platform.release(),
        'python_version': platform.python_version(),
    }


@dataclass
class CacheKeyV2:
    """Cache key based on model path (not family/size/version)."""
    model_path: str          # LM Studio's path field - unique per model file
    model_key: str           # LM Studio's modelKey for reference
    prompt_hash: str         # Hash of prompt content BEFORE modifications
    input_hash: str
    temperature: float
    max_tokens: int
    top_p: float
    context_length: Optional[int] = None
    quantization_name: Optional[str] = None  # Quantization (e.g., Q8_0, Q4_K_M)
    
    def get_cache_id(self) -> str:
        """Generate unique cache ID.
        
        Uses model_path + quantization as primary identifier to handle:
        - Publisher
        - Model name
        - Quantization level (explicit, handles multi-quant folders)
        - Specific file
        """
        key_components = [
            self.model_path,
            self.quantization_name or "unknown",  # Include quantization in cache key
            self.prompt_hash,
            self.input_hash,
            str(self.temperature),
            str(self.max_tokens),
            str(self.top_p),
            str(self.context_length) if self.context_length else "default"
        ]
        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()


@dataclass
class CachedResultV2:
    """A cached result with metadata."""
    cache_id: str
    input_text: str
    raw_response: Dict[str, Any]
    parsed_result: Optional[Dict[str, Any]]
    status_type: str         # "ok", "parse_fail", "api_error", "timeout"
    error_details: Optional[str]
    processing_time: float
    created_at: str
    replicate_index: int = 0
    parser_version: str = "v1"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResultCacheV2:
    """Cache V2 with path-based model identification."""
    
    def __init__(self, cache_dir: str = "cache_v2_test", read_only: bool = False):
        """Initialize the cache.
        
        Args:
            cache_dir: Directory containing the cache database
            read_only: If True, do not write anything to the database (for analysis)
        """
        self.cache_dir = Path(cache_dir)
        self.db_path = self.cache_dir / "results.db"
        self._read_only = read_only
        
        if not read_only:
            self.cache_dir.mkdir(exist_ok=True)
            self._init_database()
        
        # Cache LM Studio model data - skip if read_only (not needed for analysis)
        self._lms_models = {} if read_only else get_lm_studio_models()
        
        # Cache for prompt hashes
        self._prompt_hash_cache = {}
        
        # Execution context - skip write if read_only
        self._context = get_execution_context()
        self._context_id = self._get_or_create_context_id() if not read_only else None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        mode_str = "READ-ONLY" if read_only else "READ-WRITE"
        self.logger.debug(f"CacheV2 initialized at {self.db_path} ({mode_str})")
        if not read_only:
            self.logger.debug(f"Found {len(self._lms_models)} models in LM Studio")
    
    def _init_database(self):
        """Initialize SQLite database with V2 schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Model files table (indexed by path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_files (
                    model_path TEXT PRIMARY KEY,
                    model_key TEXT NOT NULL,
                    display_name TEXT,
                    architecture TEXT,
                    params_string TEXT,
                    format TEXT,
                    quantization_name TEXT,
                    quantization_bits INTEGER,
                    publisher TEXT,
                    size_bytes INTEGER,
                    max_context_length INTEGER,
                    first_seen TEXT NOT NULL
                )
            """)
            
            # Prompts table (deduplicated by hash)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    prompt_hash TEXT PRIMARY KEY,
                    prompt_name TEXT,
                    prompt_content TEXT NOT NULL,
                    source_file_path TEXT,
                    first_seen TEXT NOT NULL
                )
            """)
            
            # Input texts table (unchanged from v1)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS input_texts (
                    input_hash TEXT PRIMARY KEY,
                    input_text TEXT NOT NULL,
                    first_seen TEXT NOT NULL
                )
            """)
            
            # Execution contexts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_contexts (
                    context_id TEXT PRIMARY KEY,
                    hostname TEXT NOT NULL,
                    os_type TEXT,
                    os_version TEXT,
                    python_version TEXT,
                    lm_studio_version TEXT,
                    first_seen TEXT NOT NULL
                )
            """)
            
            # Cache keys table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_keys (
                    cache_id TEXT PRIMARY KEY,
                    model_path TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    max_tokens INTEGER NOT NULL,
                    top_p REAL NOT NULL,
                    context_length INTEGER,
                    prompt_suffix_applied TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (model_path) REFERENCES model_files(model_path),
                    FOREIGN KEY (prompt_hash) REFERENCES prompts(prompt_hash),
                    FOREIGN KEY (input_hash) REFERENCES input_texts(input_hash)
                )
            """)
            
            # Results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cached_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_id TEXT NOT NULL,
                    context_id TEXT,
                    raw_response TEXT NOT NULL,
                    parsed_result TEXT,
                    parser_version TEXT DEFAULT 'v1',
                    status_type TEXT NOT NULL,
                    error_details TEXT,
                    processing_time REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    replicate_index INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (cache_id) REFERENCES cache_keys(cache_id),
                    FOREIGN KEY (context_id) REFERENCES execution_contexts(context_id)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_model ON cache_keys(model_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_prompt ON cache_keys(prompt_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_results_cache ON cached_results(cache_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_results_status ON cached_results(status_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_key ON model_files(model_key)")
            
            conn.commit()
    
    def _get_or_create_context_id(self) -> str:
        """Get or create execution context ID."""
        ctx = self._context
        context_id = hashlib.sha256(
            f"{ctx['hostname']}|{ctx['os_type']}|{ctx['os_version']}".encode()
        ).hexdigest()[:16]
        
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO execution_contexts
                (context_id, hostname, os_type, os_version, python_version, first_seen)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (context_id, ctx['hostname'], ctx['os_type'], ctx['os_version'],
                  ctx['python_version'], now))
            conn.commit()
        
        return context_id
    
    def _ensure_model_registered(self, model_key: str, expected_quantization: Optional[str] = None, read_only: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """Ensure model is registered in model_files table. Returns (model_path, quantization_name).
        
        IMPORTANT: Always checks LM Studio first to get current quantization (critical for
        multi-quant folders where same model_key can have different quantizations loaded).
        Only falls back to database when LM Studio is unavailable.
        
        Args:
            model_key: LM Studio model key
            expected_quantization: Expected quantization (e.g., 'Q8_0'). If provided,
                will validate that LM Studio has the correct quantization loaded.
            read_only: If True, only read from database, do not modify (for analysis/retrieval)
        
        Raises:
            ValueError: If expected_quantization is provided and doesn't match actual
        """
        # For read-only operations, skip LM Studio and use database directly
        # This prevents overwriting cached metadata with whatever is currently loaded
        if read_only:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT model_path, quantization_name FROM model_files WHERE model_key = ?", (model_key,)
                )
                row = cursor.fetchone()
                if row:
                    self.logger.debug(f"Model {model_key} found in database (read-only mode)")
                    return row[0], row[1]
            self.logger.warning(f"Model {model_key} not found in database (read-only mode)")
            return None, None
        
        # Use cached LM Studio models (loaded once at initialization)
        if model_key in self._lms_models:
            # LM Studio is available - use current model info (includes current quantization)
            model_info = self._lms_models[model_key]
            model_path = model_info['path']
            now = datetime.now().isoformat()
            
            quant = model_info.get('quantization', {})
            quant_name = quant.get('name') if isinstance(quant, dict) else None
            quant_bits = quant.get('bits') if isinstance(quant, dict) else None
            
            # VALIDATE QUANTIZATION MISMATCH
            if expected_quantization and quant_name and expected_quantization != quant_name:
                raise ValueError(
                    f"❌ QUANTIZATION MISMATCH for {model_key}!\n"
                    f"   Expected: {expected_quantization}\n"
                    f"   Actually loaded in LM Studio: {quant_name}\n"
                    f"   Model path: {model_path}\n\n"
                    f"   Please load the correct quantization in LM Studio before running.\n"
                    f"   This check prevents accidentally caching results from the wrong quantization."
                )
            
            self.logger.debug(f"Model {model_key} from LM Studio: path={model_path}, quant={quant_name}")
            if expected_quantization:
                self.logger.info(f"✓ Quantization verified: {quant_name} matches expected {expected_quantization}")
            
            with sqlite3.connect(self.db_path) as conn:
                # Use REPLACE to update metadata if model is re-run with different quantization
                conn.execute("""
                    INSERT OR REPLACE INTO model_files
                    (model_path, model_key, display_name, architecture, params_string,
                     format, quantization_name, quantization_bits, publisher, size_bytes,
                     max_context_length, first_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_path,
                    model_key,
                    model_info.get('displayName'),
                    model_info.get('architecture'),
                    model_info.get('paramsString'),
                    model_info.get('format'),
                    quant_name,
                    quant_bits,
                    model_info.get('publisher'),
                    model_info.get('sizeBytes'),
                    model_info.get('maxContextLength'),
                    now
                ))
                conn.commit()
            
            return model_path, quant_name
        
        # LM Studio not available - fall back to database (offline mode)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT model_path, quantization_name FROM model_files WHERE model_key = ?", (model_key,)
            )
            row = cursor.fetchone()
            if row:
                self.logger.debug(f"Model {model_key} found in database (offline mode)")
                return row[0], row[1]
        
        self.logger.warning(f"Model {model_key} not found in LM Studio or database")
        return None, None
    
    def _ensure_prompt_registered(self, prompt_content: str, prompt_name: str = None,
                                   source_file: str = None) -> str:
        """Ensure prompt is registered. Returns prompt_hash."""
        if prompt_content not in self._prompt_hash_cache:
            self._prompt_hash_cache[prompt_content] = hashlib.sha256(
                prompt_content.encode()
            ).hexdigest()
        
        prompt_hash = self._prompt_hash_cache[prompt_content]
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO prompts
                (prompt_hash, prompt_name, prompt_content, source_file_path, first_seen)
                VALUES (?, ?, ?, ?, ?)
            """, (prompt_hash, prompt_name, prompt_content, source_file, now))
            conn.commit()
        
        return prompt_hash
    
    def _ensure_input_registered(self, input_text: str) -> str:
        """Ensure input is registered. Returns input_hash."""
        input_hash = hashlib.md5(input_text.encode()).hexdigest()
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO input_texts
                (input_hash, input_text, first_seen)
                VALUES (?, ?, ?)
            """, (input_hash, input_text, now))
            conn.commit()
        
        return input_hash
    
    def create_cache_key(self, model_key: str, prompt_content: str, input_text: str,
                        temperature: float, max_tokens: int, top_p: float,
                        context_length: Optional[int] = None,
                        prompt_name: str = None,
                        prompt_suffix: str = None,
                        expected_quantization: Optional[str] = None,
                        read_only: bool = False) -> Optional[CacheKeyV2]:
        """Create a cache key from experiment parameters.
        
        Args:
            expected_quantization: Expected quantization (e.g., 'Q8_0'). If provided,
                will validate that LM Studio has the correct quantization loaded.
                Raises ValueError if mismatch detected.
            read_only: If True, only read from database, do not modify model_files table.
        """
        model_path, quantization_name = self._ensure_model_registered(model_key, expected_quantization, read_only=read_only)
        if not model_path:
            return None
        
        prompt_hash = self._ensure_prompt_registered(prompt_content, prompt_name)
        input_hash = self._ensure_input_registered(input_text)
        
        return CacheKeyV2(
            model_path=model_path,
            model_key=model_key,
            prompt_hash=prompt_hash,
            input_hash=input_hash,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            context_length=context_length,
            quantization_name=quantization_name
        )
    
    def check_cache(self, cache_key: CacheKeyV2, num_replicates: int = 1,
                   include_errors: bool = True) -> Tuple[List[CachedResultV2], int]:
        """Check if results exist in cache."""
        cache_id = cache_key.get_cache_id()
        
        status_filter = "" if include_errors else "AND status_type NOT IN ('api_error', 'timeout')"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"""
                SELECT COUNT(*) FROM cached_results WHERE cache_id = ? {status_filter}
            """, (cache_id,))
            
            existing_count = cursor.fetchone()[0]
            missing_count = max(0, num_replicates - existing_count)
            
            existing_results = []
            if existing_count > 0:
                # Get input_text for results
                cursor = conn.execute("""
                    SELECT it.input_text FROM cache_keys ck
                    JOIN input_texts it ON ck.input_hash = it.input_hash
                    WHERE ck.cache_id = ?
                """, (cache_id,))
                row = cursor.fetchone()
                input_text = row[0] if row else ""
                
                cursor = conn.execute(f"""
                    SELECT raw_response, parsed_result, status_type, error_details,
                           processing_time, created_at, replicate_index, parser_version
                    FROM cached_results 
                    WHERE cache_id = ? {status_filter}
                    ORDER BY replicate_index
                """, (cache_id,))
                
                for row in cursor.fetchall():
                    result = CachedResultV2(
                        cache_id=cache_id,
                        input_text=input_text,
                        raw_response=json.loads(row[0]),
                        parsed_result=json.loads(row[1]) if row[1] else None,
                        status_type=row[2],
                        error_details=row[3],
                        processing_time=row[4],
                        created_at=row[5],
                        replicate_index=row[6],
                        parser_version=row[7] or "v1"
                    )
                    existing_results.append(result)
        
        return existing_results, missing_count
    
    def store_result(self, cache_key: CacheKeyV2, input_text: str,
                    raw_response: Dict[str, Any], parsed_result: Optional[Dict[str, Any]],
                    status_type: str, processing_time: float,
                    replicate_index: int = 0,
                    error_details: str = None,
                    prompt_suffix: str = None) -> str:
        """Store a result in the cache."""
        cache_id = cache_key.get_cache_id()
        created_at = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Ensure cache key exists (store quantization for queryability)
            conn.execute("""
                INSERT OR IGNORE INTO cache_keys
                (cache_id, model_path, prompt_hash, input_hash, temperature,
                 max_tokens, top_p, context_length, prompt_suffix_applied, created_at, quantization_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_id, cache_key.model_path, cache_key.prompt_hash, cache_key.input_hash,
                cache_key.temperature, cache_key.max_tokens, cache_key.top_p,
                cache_key.context_length, prompt_suffix, created_at, cache_key.quantization_name
            ))
            
            # Store result
            conn.execute("""
                INSERT INTO cached_results
                (cache_id, context_id, raw_response, parsed_result, parser_version,
                 status_type, error_details, processing_time, created_at, replicate_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_id, self._context_id,
                json.dumps(raw_response),
                json.dumps(parsed_result) if parsed_result else None,
                "v1",
                status_type,
                error_details,
                processing_time,
                created_at,
                replicate_index
            ))
            
            conn.commit()
        
        return cache_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache_keys")
            unique_entries = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM cached_results")
            total_results = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM model_files")
            model_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM prompts")
            prompt_count = cursor.fetchone()[0]
            
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                "unique_entries": unique_entries,
                "total_results": total_results,
                "model_count": model_count,
                "prompt_count": prompt_count,
                "database_size_bytes": db_size,
                "database_size_mb": db_size / (1024 * 1024),
                "database_path": str(self.db_path),
                "execution_context": self._context
            }
    
    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get model info from cache or LM Studio."""
        if model_key in self._lms_models:
            return self._lms_models[model_key]
        
        # Try to get from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM model_files WHERE model_key = ?
            """, (model_key,))
            row = cursor.fetchone()
            if row:
                cols = [desc[0] for desc in cursor.description]
                return dict(zip(cols, row))
        
        return None
    
    def load_results_for_analysis(self, model_key: str, input_texts: List[str], 
                                  prompt_content: str, prompt_name: str,
                                  temperature: float = 0.0, max_tokens: int = 256,
                                  top_p: float = 1.0, prompt_suffix: str = None,
                                  model_family: str = None, model_size: str = None,
                                  model_version: str = None) -> 'pd.DataFrame':
        """
        Load cached results for analysis given model key and list of input texts.
        
        Args:
            model_key: LM Studio model key (lm_studio_id from config)
            input_texts: List of input texts to retrieve results for
            prompt_content: Full prompt content (for calculating prompt_hash)
            prompt_name: Name of the prompt
            temperature: API temperature parameter
            max_tokens: API max_tokens parameter
            top_p: API top_p parameter
            prompt_suffix: Optional prompt suffix (e.g., '/no_think')
            model_family: Model family name (e.g., 'gemma', 'qwen')
            model_size: Model size (e.g., '270m-it', '0.6b')
            model_version: Model version (e.g., '3.0')
            
        Returns:
            DataFrame with columns: input_text, parsed fields, status, model metadata
        """
        import pandas as pd
        
        # Get model info to resolve path
        model_info = self.get_model_info(model_key)
        if not model_info:
            return pd.DataFrame()
        
        # Handle both LM Studio (uses 'path') and database (uses 'model_path')
        model_path = model_info.get('path') or model_info.get('model_path') or model_key
        
        results = []
        for input_text in input_texts:
            # Create cache key (read_only=True to prevent overwriting metadata during retrieval)
            cache_key = self.create_cache_key(
                model_key=model_key,
                prompt_content=prompt_content,
                input_text=input_text,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                prompt_name=prompt_name,
                prompt_suffix=prompt_suffix,
                expected_quantization=None,  # No validation when retrieving existing results
                read_only=True  # Don't modify database during analysis
            )
            
            if cache_key:
                cached_results, _ = self.check_cache(cache_key, num_replicates=1)
                
                for cached in cached_results:
                    # Construct model_full_name for compatibility with analysis tools
                    full_name = f"{model_family}_{model_size}_{model_version}" if model_family else model_path
                    
                    result = {
                        'input_text': input_text,
                        'cache_id': cached.cache_id,  # For traceability
                        'model_family': model_family or '',
                        'model_size': model_size or '',
                        'model_version': model_version or '',
                        'model_full_name': full_name,
                        'model_path': model_path,
                        'status': cached.status_type,
                        'raw_response': cached.raw_response,
                        'processing_time': cached.processing_time,
                        'created_at': cached.created_at,  # For traceability
                    }
                    
                    # Add parsed result fields if available
                    if cached.parsed_result:
                        for key, value in cached.parsed_result.items():
                            result[key] = value
                    
                    results.append(result)
        
        return pd.DataFrame(results)

    def generate_cache_manifest(self, results_df: 'pd.DataFrame') -> Dict[str, Any]:
        """
        Generate a cache manifest for traceability.
        
        Creates a summary of which cache entries were used in an analysis,
        allowing results to be traced back to specific cache entries.
        
        Args:
            results_df: DataFrame from load_results_for_analysis containing cache_id column
            
        Returns:
            Dict with manifest information including:
            - timestamp: When manifest was generated
            - cache_dir: Path to cache used
            - total_entries: Number of cache entries used
            - models: Summary by model
            - cache_ids: List of all cache IDs used
        """
        from datetime import datetime
        
        if 'cache_id' not in results_df.columns:
            return {
                'error': 'No cache_id column in results - cannot generate manifest',
                'timestamp': datetime.now().isoformat()
            }
        
        unique_cache_ids = results_df['cache_id'].unique().tolist()
        
        # Group by model
        models_summary = {}
        if 'model_family' in results_df.columns and 'model_size' in results_df.columns:
            for (family, size), group in results_df.groupby(['model_family', 'model_size']):
                model_key = f"{family}_{size}"
                models_summary[model_key] = {
                    'entries': len(group),
                    'cache_ids': group['cache_id'].unique().tolist(),
                    'model_path': group['model_path'].iloc[0] if 'model_path' in group.columns else None,
                    'earliest_entry': group['created_at'].min() if 'created_at' in group.columns else None,
                    'latest_entry': group['created_at'].max() if 'created_at' in group.columns else None,
                }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cache_dir': str(self.cache_dir),
            'cache_version': 2,
            'total_entries': len(unique_cache_ids),
            'total_results': len(results_df),
            'models': models_summary,
            'cache_ids': unique_cache_ids,
        }

