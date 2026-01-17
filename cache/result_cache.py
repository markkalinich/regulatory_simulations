#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced caching and replication system for experiment results.

This system provides:
1. Intelligent caching to avoid redundant API calls
2. Support for replicates (multiple runs of identical conditions)
3. Cache invalidation and management
4. Result deduplication and retrieval
"""

import hashlib
import json
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

from orchestration.experiment_manager import ExperimentConfig


@dataclass
class CacheKey:
    """Unique identifier for a cached result."""
    model_family: str
    model_size: str
    model_version: str
    model_full_name: str
    prompt_name: str  # Kept for metadata/display only, NOT used in cache key
    prompt_version: str
    prompt_hash: str  # Hash of prompt content to detect changes
    input_text: str
    input_hash: str   # Hash of input text for efficiency
    temperature: float
    max_tokens: int
    top_p: float
    
    def get_cache_id(self) -> str:
        """Generate unique cache ID from key components.
        
        NOTE: prompt_name is intentionally EXCLUDED from cache key.
        Cache hits are based on prompt CONTENT (prompt_hash), not arbitrary names.
        This prevents cache misses when the same prompt is used with different names.
        """
        key_components = [
            self.model_family, self.model_size, self.model_version,
            # prompt_name intentionally excluded - use prompt_hash for content-based caching
            self.prompt_hash,
            self.input_hash, str(self.temperature), str(self.max_tokens), str(self.top_p)
        ]
        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()


@dataclass
class CachedResult:
    """A cached result with metadata."""
    cache_id: str
    input_text: str
    raw_response: Dict[str, Any]
    parsed_result: Optional[Dict[str, Any]]
    status: str
    processing_time: float
    created_at: str
    replicate_index: int = 0  # For tracking multiple runs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedResult':
        """Create from dictionary."""
        return cls(**data)


class ResultCache:
    """Cache for experiment results with replication support."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize the cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / "results.db"
        self._init_database()
        
        # Cache for prompt content hashes to avoid repeated computation
        self._prompt_hash_cache = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Cache keys table (optimized for fast lookups)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_keys (
                    cache_id TEXT PRIMARY KEY,
                    model_family TEXT NOT NULL,
                    model_size TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    model_full_name TEXT NOT NULL,
                    prompt_name TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    max_tokens INTEGER NOT NULL,
                    top_p REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Separate table for input text lookup (queryable but not in main cache path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS input_texts (
                    input_hash TEXT PRIMARY KEY,
                    input_text TEXT NOT NULL,
                    first_seen TEXT NOT NULL
                )
            """)
            
            # Results table (supports multiple replicates per cache_id)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cached_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_id TEXT NOT NULL,
                    raw_response TEXT NOT NULL,
                    parsed_result TEXT,
                    status TEXT NOT NULL,
                    processing_time REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    replicate_index INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (cache_id) REFERENCES cache_keys (cache_id)
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_id ON cached_results (cache_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_input_hash ON cache_keys (input_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_prompt ON cache_keys (model_family, model_size, prompt_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_input_text_search ON input_texts (input_text)")
            
            conn.commit()
    
    def _create_cache_key(self, config: ExperimentConfig, input_text: str, prompt_content: str) -> CacheKey:
        """Create a cache key from experiment configuration and input."""
        
        # Hash the input text (always different)
        input_hash = hashlib.md5(input_text.encode()).hexdigest()
        
        # Cache prompt hash since it's the same for all inputs in a batch
        if prompt_content not in self._prompt_hash_cache:
            self._prompt_hash_cache[prompt_content] = hashlib.md5(prompt_content.encode()).hexdigest()
        prompt_hash = self._prompt_hash_cache[prompt_content]
        
        return CacheKey(
            model_family=config.model.family,
            model_size=config.model.size,
            model_version=config.model.version,
            model_full_name=config.model.full_name,
            prompt_name=config.prompt.name,
            prompt_version=config.prompt.version,
            prompt_hash=prompt_hash,
            input_text="",  # Don't store full text - use hash for lookups
            input_hash=input_hash,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p
        )
    
    def check_cache(self, config: ExperimentConfig, input_text: str, prompt_content: str, 
                   num_replicates: int = 1, include_errors: bool = True) -> Tuple[List[CachedResult], int]:
        """
        Check if results exist in cache.
        
        Args:
            config: Experiment configuration
            input_text: Input text to check
            prompt_content: Full prompt content
            num_replicates: Number of replicates needed
            include_errors: If True, return all entries including api_errors (for reproducibility).
                           If False, exclude api_error entries (for retry logic).
            
        Returns:
            Tuple of (existing_results, missing_count)
        """
        cache_key = self._create_cache_key(config, input_text, prompt_content)
        cache_id = cache_key.get_cache_id()
        
        # Build status filter clause
        status_filter = "" if include_errors else "AND status NOT LIKE 'api_error%'"
        
        with sqlite3.connect(self.db_path) as conn:
            # First, just count existing results - much faster
            cursor = conn.execute(f"""
                SELECT COUNT(*) FROM cached_results WHERE cache_id = ? {status_filter}
            """, (cache_id,))
            
            existing_count = cursor.fetchone()[0]
            missing_count = max(0, num_replicates - existing_count)
            
            # Only fetch full results if we need them (for reuse)
            existing_results = []
            if existing_count > 0:
                cursor = conn.execute(f"""
                    SELECT raw_response, parsed_result, status, processing_time, 
                           created_at, replicate_index
                    FROM cached_results 
                    WHERE cache_id = ? {status_filter}
                    ORDER BY replicate_index
                """, (cache_id,))
                
                rows = cursor.fetchall()
                
                for row in rows:
                    # Lazy JSON parsing - only parse when needed
                    result = CachedResult(
                        cache_id=cache_id,
                        input_text=input_text,
                        raw_response=json.loads(row[0]),
                        parsed_result=json.loads(row[1]) if row[1] else None,
                        status=row[2],
                        processing_time=row[3],
                        created_at=row[4],
                        replicate_index=row[5]
                    )
                    existing_results.append(result)
        
        return existing_results, missing_count

    def store_result(self, config: ExperimentConfig, input_text: str, prompt_content: str,
                    raw_response: Dict[str, Any], parsed_result: Optional[Dict[str, Any]],
                    status: str, processing_time: float, replicate_index: int = 0) -> str:
        """
        Store a result in the cache.
        
        Args:
            config: Experiment configuration
            input_text: Input text
            prompt_content: Full prompt content
            raw_response: Raw API response
            parsed_result: Parsed result
            status: Processing status
            processing_time: Time taken to process
            replicate_index: Index for this replicate (0-based)
            
        Returns:
            Cache ID of stored result
        """
        cache_key = self._create_cache_key(config, input_text, prompt_content)
        cache_id = cache_key.get_cache_id()
        created_at = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Store input text for queryability (separate table, not in main cache path)
            conn.execute("""
                INSERT OR IGNORE INTO input_texts (input_hash, input_text, first_seen)
                VALUES (?, ?, ?)
            """, (cache_key.input_hash, input_text, created_at))
            
            # Insert cache key if not exists (optimized for fast lookups)
            conn.execute("""
                INSERT OR IGNORE INTO cache_keys (
                    cache_id, model_family, model_size, model_version, model_full_name,
                    prompt_name, prompt_version, prompt_hash, input_hash,
                    temperature, max_tokens, top_p, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_id, cache_key.model_family, cache_key.model_size, cache_key.model_version,
                cache_key.model_full_name, cache_key.prompt_name, cache_key.prompt_version,
                cache_key.prompt_hash, cache_key.input_hash,
                cache_key.temperature, cache_key.max_tokens, cache_key.top_p, created_at
            ))
            
            # Insert result
            conn.execute("""
                INSERT INTO cached_results (
                    cache_id, raw_response, parsed_result, status, 
                    processing_time, created_at, replicate_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_id,
                json.dumps(raw_response),
                json.dumps(parsed_result) if parsed_result else None,
                status,
                processing_time,
                created_at,
                replicate_index
            ))
            
            conn.commit()
        
        return cache_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Count unique entries and total results
            cursor = conn.execute("SELECT COUNT(*) FROM cache_keys")
            unique_entries = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM cached_results")
            total_results = cursor.fetchone()[0]
            
            # Get database size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                "unique_entries": unique_entries,
                "total_results": total_results,
                "database_size_bytes": db_size,
                "database_size_mb": db_size / (1024 * 1024),
                "database_path": str(self.db_path)
            }
    
    def count_api_errors(self, model_family: str, model_size: str, prompt_name: str) -> int:
        """
        Count API errors for a specific model/prompt combination.
        
        Args:
            model_family: Model family (e.g., 'gemma', 'mental_health')
            model_size: Model size (e.g., '2b_v1', '27b-it')
            prompt_name: Prompt name (e.g., 'system_suicide_detection_v2')
            
        Returns:
            Count of API error entries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*)
                FROM cached_results cr
                JOIN cache_keys ck ON cr.cache_id = ck.cache_id
                WHERE ck.model_family = ?
                AND ck.model_size = ?
                AND ck.prompt_name = ?
                AND cr.status LIKE 'api_error%'
            """, (model_family, model_size, prompt_name))
            return cursor.fetchone()[0]
    
    def count_valid_results(self, config: ExperimentConfig, input_text: str, 
                            prompt_content: str) -> int:
        """
        Count valid (non-error) cached results for a specific input.
        
        This excludes api_error entries, giving an accurate count of
        results that won't need to be re-run.
        """
        cache_key = self._create_cache_key(config, input_text, prompt_content)
        cache_id = cache_key.get_cache_id()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM cached_results 
                WHERE cache_id = ? AND status NOT LIKE 'api_error%'
            """, (cache_id,))
            return cursor.fetchone()[0]
    
    def delete_api_errors(self, model_family: str, model_size: str, prompt_name: str) -> int:
        """
        Delete API error entries for a specific model/prompt combination.
        This allows re-running failed API calls.
        
        Args:
            model_family: Model family (e.g., 'gemma', 'mental_health')
            model_size: Model size (e.g., '2b_v1', '27b-it')
            prompt_name: Prompt name (e.g., 'system_suicide_detection_v2')
            
        Returns:
            Number of deleted entries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM cached_results 
                WHERE cache_id IN (
                    SELECT cache_id FROM cache_keys 
                    WHERE model_family = ?
                    AND model_size = ?
                    AND prompt_name = ?
                )
                AND status LIKE 'api_error%'
            """, (model_family, model_size, prompt_name))
            deleted_count = cursor.rowcount
            conn.commit()
            
            if deleted_count > 0:
                self.logger.info(f"Deleted {deleted_count} API error entries for {model_family}/{model_size}")
            
            return deleted_count
    
    def clear_cache(self, confirm: bool = False):
        """Clear all cached results."""
        if not confirm:
            raise ValueError("Must set confirm=True to clear cache")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cached_results")
            conn.execute("DELETE FROM cache_keys")
            conn.execute("DELETE FROM input_texts")
            conn.commit()
        
        self.logger.info("Cache cleared successfully")
    
    def find_results_by_input_text(self, input_text: str) -> List[CachedResult]:
        """
        Find all cached results for a specific input text.
        
        Args:
            input_text: The exact input text to search for
            
        Returns:
            List of cached results for this input text
        """
        input_hash = hashlib.md5(input_text.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT cr.cache_id, cr.raw_response, cr.parsed_result, cr.status, 
                       cr.processing_time, cr.created_at, cr.replicate_index
                FROM cached_results cr
                JOIN cache_keys ck ON cr.cache_id = ck.cache_id
                WHERE ck.input_hash = ?
                ORDER BY cr.created_at, cr.replicate_index
            """, (input_hash,))
            
            results = []
            for row in cursor.fetchall():
                result = CachedResult(
                    cache_id=row[0],
                    input_text=input_text,
                    raw_response=json.loads(row[1]),
                    parsed_result=json.loads(row[2]) if row[2] else None,
                    status=row[3],
                    processing_time=row[4],
                    created_at=row[5],
                    replicate_index=row[6]
                )
                results.append(result)
            
            return results
    
    def search_input_texts(self, search_term: str, limit: int = 50) -> List[str]:
        """
        Search for input texts containing a specific term.
        
        Args:
            search_term: Term to search for in input texts
            limit: Maximum number of results to return
            
        Returns:
            List of input texts containing the search term
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT input_text 
                FROM input_texts 
                WHERE input_text LIKE ? 
                ORDER BY first_seen DESC
                LIMIT ?
            """, (f"%{search_term}%", limit))
            
            return [row[0] for row in cursor.fetchall()]
    
    def get_all_input_texts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all cached input texts with metadata.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with input text and metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT it.input_text, it.first_seen, COUNT(DISTINCT ck.cache_id) as config_count,
                       COUNT(cr.id) as total_results
                FROM input_texts it
                JOIN cache_keys ck ON it.input_hash = ck.input_hash
                JOIN cached_results cr ON ck.cache_id = cr.cache_id
                GROUP BY it.input_hash, it.input_text, it.first_seen
                ORDER BY it.first_seen DESC
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'input_text': row[0],
                    'first_seen': row[1],
                    'config_count': row[2],  # How many different model/prompt configs
                    'total_results': row[3]  # Total results across all configs/replicates
                })
            
            return results
    
    def get_all_cached_results_dataframe(self, experiment_name_pattern: Optional[str] = None) -> pd.DataFrame:
        """
        Get all cached results as a pandas DataFrame for analysis.
        
        Args:
            experiment_name_pattern: Optional pattern to filter by prompt name (e.g., 'system_suicide_detection_v2')
            
        Returns:
            DataFrame with all cached results and metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            # Build the query with optional filtering
            query = """
                SELECT 
                    ck.model_family,
                    ck.model_size,
                    ck.model_version,
                    ck.model_full_name,
                    ck.prompt_name,
                    ck.prompt_version,
                    ck.temperature,
                    ck.max_tokens,
                    ck.top_p,
                    it.input_text,
                    cr.raw_response,
                    cr.parsed_result,
                    cr.status,
                    cr.processing_time,
                    cr.created_at,
                    cr.replicate_index
                FROM cache_keys ck
                JOIN cached_results cr ON ck.cache_id = cr.cache_id
                JOIN input_texts it ON ck.input_hash = it.input_hash
            """
            
            params = []
            if experiment_name_pattern:
                query += " WHERE ck.prompt_name LIKE ?"
                params.append(f"%{experiment_name_pattern}%")
            
            query += " ORDER BY ck.model_family, ck.model_size, it.input_text, cr.replicate_index"
            
            # Execute query and convert to DataFrame
            df = pd.read_sql_query(query, conn, params=params)
            
            # Parse JSON columns
            if len(df) > 0:
                df['raw_response'] = df['raw_response'].apply(lambda x: json.loads(x) if x else {})
                df['parsed_result'] = df['parsed_result'].apply(lambda x: json.loads(x) if x else None)
                
                # Extract commonly used fields from parsed_result for easier analysis
                if 'parsed_result' in df.columns:
                    df['safety_type'] = df['parsed_result'].apply(
                        lambda x: x.get('safety_type') if isinstance(x, dict) else None
                    )
                    df['prior_safety_type'] = df['input_text'].apply(
                        lambda x: self._extract_safety_type_from_input(x)
                    )
            
            return df
    
    def _extract_safety_type_from_input(self, input_text: str) -> str:
        """
        Extract the prior safety type from input text.
        This is a helper method that should match the logic used in your experiment setup.
        """
        # This should match your input data structure
        # You may need to adjust this based on how your input data is formatted
        try:
            # If input is JSON-like, try to parse it
            if input_text.strip().startswith('{'):
                import json
                parsed = json.loads(input_text)
                return parsed.get('safety_type', 'unknown')
            else:
                # For plain text, you might need different logic
                # This is a placeholder - adjust based on your data format
                return 'unknown'
        except:
            return 'unknown'
    
    def load_results_for_analysis(self, config: ExperimentConfig, input_texts: List[str], 
                                  prompt_content: str) -> pd.DataFrame:
        """
        Load cached results for analysis given model config and list of input texts.
        This is the PRIMARY method for loading data for analysis - ensures single source of truth from cache.
        
        Args:
            config: Experiment configuration (model, prompt, API params)
            input_texts: List of input texts to retrieve results for
            prompt_content: Full prompt content (for calculating prompt_hash)
            
        Returns:
            DataFrame with columns: input_text, safety_type, counseling_request, therapy_request, 
                                   therapy_engagement, status, confidences, model metadata
        """
        # CRITICAL: Query by cache_id which is computed from prompt_hash (content-based)
        # This ensures cache hits are based on prompt CONTENT, not arbitrary prompt names.
        # The cache_id incorporates: model_family, model_size, model_version, prompt_hash,
        # input_hash, temperature, max_tokens, top_p
        
        # Pre-compute all cache IDs for this batch
        cache_keys_map = {}
        for input_text in input_texts:
            cache_key = self._create_cache_key(config, input_text, prompt_content)
            cache_id = cache_key.get_cache_id()
            cache_keys_map[input_text] = cache_id
        
        # Single query to get all results by cache_id
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join('?' * len(cache_keys_map))
            query = f"""
                SELECT 
                    ck.cache_id,
                    it.input_text,
                    cr.parsed_result,
                    cr.status,
                    cr.processing_time,
                    cr.replicate_index,
                    ck.model_family,
                    ck.model_size,
                    ck.model_version,
                    ck.model_full_name,
                    ck.prompt_name,
                    ck.prompt_version,
                    ck.temperature,
                    ck.max_tokens,
                    cr.created_at
                FROM cache_keys ck
                JOIN cached_results cr ON ck.cache_id = cr.cache_id
                JOIN input_texts it ON ck.input_hash = it.input_hash
                WHERE ck.cache_id IN ({placeholders})
                ORDER BY it.input_text, cr.replicate_index
            """
            
            cursor = conn.execute(query, list(cache_keys_map.values()))
            rows = cursor.fetchall()
        
        # Convert to DataFrame
        if not rows:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'input_text', 'safety_type', 'counseling_request', 'therapy_request', 'therapy_engagement',
                'safety_type_confidence', 'counseling_request_confidence', 
                'therapy_request_confidence', 'therapy_engagement_confidence',
                'status', 'processing_time', 'replicate_index',
                'model_family', 'model_size', 'model_version', 'model_full_name',
                'prompt_name', 'prompt_version', 'temperature', 'max_tokens', 'created_at'
            ])
        
        # Parse rows into records
        records = []
        for row in rows:
            parsed_result = json.loads(row[2]) if row[2] else {}
            
            record = {
                'input_text': row[1],
                'safety_type': parsed_result.get('safety_type'),
                'counseling_request': parsed_result.get('counseling_request'),
                'therapy_request': parsed_result.get('therapy_request'),
                'therapy_engagement': parsed_result.get('therapy_engagement'),
                'safety_type_confidence': parsed_result.get('safety_type_confidence'),
                'counseling_request_confidence': parsed_result.get('counseling_request_confidence'),
                'therapy_request_confidence': parsed_result.get('therapy_request_confidence'),
                'therapy_engagement_confidence': parsed_result.get('therapy_engagement_confidence'),
                'status': row[3],
                'processing_time': row[4],
                'replicate_index': row[5],
                'model_family': row[6],
                'model_size': row[7],
                'model_version': row[8],
                'model_full_name': row[9],
                'prompt_name': row[10],
                'prompt_version': row[11],
                'temperature': row[12],
                'max_tokens': row[13],
                'created_at': row[14]
            }
            records.append(record)
        
        return pd.DataFrame(records)