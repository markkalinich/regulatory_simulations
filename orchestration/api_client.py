#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API client for LM Studio server interactions.
"""
import requests
import aiohttp
import asyncio
import time
from typing import Dict, Any, Optional
from config import EvaluationConfig


class LMStudioAPIError(Exception):
    """Custom exception for LM Studio API errors."""
    pass


class ModelNotLoadedError(LMStudioAPIError):
    """Exception raised when model is not loaded."""
    pass


class LMStudioClient:
    """Client for interacting with LM Studio API."""
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize the API client.
        
        Args:
            config: Configuration object with API settings
        """
        self.config = config
        self.base_url = config.base_url.rstrip('/')
        
    def call_chat_completion(self, system_prompt: str, text: str) -> Dict[str, Any]:
        """
        Make a chat completion request to the LM Studio API.
        
        Args:
            system_prompt: System prompt to guide model behavior
            text: User text to process
            
        Returns:
            Full API response as dictionary
            
        Raises:
            LMStudioAPIError: If API request fails
        """
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "stream": False,
        }
        
        try:
            response = requests.post(
                url, 
                json=payload, 
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise LMStudioAPIError(f"API request failed: {e}") from e
        except ValueError as e:
            raise LMStudioAPIError(f"Invalid JSON response: {e}") from e
    
    async def call_chat_completion_async(self, system_prompt: str, text: str) -> Dict[str, Any]:
        """
        Make an async chat completion request to the LM Studio API.
        
        Args:
            system_prompt: System prompt to guide model behavior
            text: User text to process
            
        Returns:
            Full API response as dictionary
            
        Raises:
            LMStudioAPIError: If API request fails
        """
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "stream": False,
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LMStudioAPIError(f"API request failed with status {response.status}: {error_text}")
                    return await response.json()
                    
        except aiohttp.ClientError as e:
            raise LMStudioAPIError(f"API request failed: {e}") from e
        except asyncio.TimeoutError as e:
            raise LMStudioAPIError(f"Request timeout: {e}") from e
        except ValueError as e:
            raise LMStudioAPIError(f"Invalid JSON response: {e}") from e

    def extract_content(self, response: Dict[str, Any]) -> str:
        """
        Extract message content from API response.
        
        Args:
            response: Full API response dictionary
            
        Returns:
            Message content string, or empty string if not found
        """
        try:
            return (response
                   .get("choices", [{}])[0]
                   .get("message", {})
                   .get("content", ""))
        except (IndexError, KeyError, TypeError):
            return ""
    
    def warmup_model(self, max_retries: int = 3, retry_delay: float = 2.0) -> bool:
        """
        Ensure model is loaded by sending a simple warmup request.
        
        This sends a minimal request to force LM Studio to load the model
        before we start the main batch of requests.
        
        Args:
            max_retries: Maximum number of warmup attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            True if model is confirmed loaded, False otherwise
        """
        warmup_prompt = "Respond with only the word 'ready'."
        warmup_text = "Are you ready?"
        
        for attempt in range(max_retries):
            try:
                print(f"  🔄 Warming up model {self.config.model}... (attempt {attempt + 1}/{max_retries})")
                start_time = time.time()
                
                response = self.call_chat_completion(warmup_prompt, warmup_text)
                
                elapsed = time.time() - start_time
                content = self.extract_content(response)
                
                if response and "choices" in response:
                    print(f"  ✅ Model loaded and responding ({elapsed:.1f}s)")
                    return True
                    
            except LMStudioAPIError as e:
                error_msg = str(e)
                if "Model does not exist" in error_msg or "Failed to load model" in error_msg:
                    print(f"  ⏳ Model not loaded yet, waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"  ⚠️ Warmup error: {error_msg[:100]}")
                    time.sleep(retry_delay)
        
        print(f"  ❌ Failed to warm up model after {max_retries} attempts")
        return False
    
    async def warmup_model_async(self, max_retries: int = 3, retry_delay: float = 2.0) -> bool:
        """
        Async version of warmup_model.
        """
        warmup_prompt = "Respond with only the word 'ready'."
        warmup_text = "Are you ready?"
        
        for attempt in range(max_retries):
            try:
                print(f"  🔄 Warming up model {self.config.model}... (attempt {attempt + 1}/{max_retries})")
                start_time = time.time()
                
                response = await self.call_chat_completion_async(warmup_prompt, warmup_text)
                
                elapsed = time.time() - start_time
                
                if response and "choices" in response:
                    print(f"  ✅ Model loaded and responding ({elapsed:.1f}s)")
                    return True
                    
            except LMStudioAPIError as e:
                error_msg = str(e)
                if "Model does not exist" in error_msg or "Failed to load model" in error_msg:
                    print(f"  ⏳ Model not loaded yet, waiting {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"  ⚠️ Warmup error: {error_msg[:100]}")
                    await asyncio.sleep(retry_delay)
        
        print(f"  ❌ Failed to warm up model after {max_retries} attempts")
        return False