#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema validation and data coercion for LM evaluation responses.
"""
import json
from typing import Dict, Any, Optional
from config.constants import SAFETY_TYPES, THERAPY_REQUESTS, THERAPY_ENGAGEMENT

# Valid counseling request categories for suicidal ideation detection task
COUNSELING_REQUESTS = [
    "no_request",
    "clear_request", 
    "ambiguous_request",
    "non_counseling_question"
]


def extract_first_json_object(s: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first top-level JSON object from a string using stack-based parsing.
    More robust than regex-based approaches.
    
    Args:
        s: String that may contain JSON object(s)
        
    Returns:
        First valid JSON object found, or None if none found
    """
    start_idx = s.find("{")
    if start_idx == -1:
        return None
        
    depth = 0
    for i in range(start_idx, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start_idx:i+1])
                except json.JSONDecodeError:
                    return None
    return None


def validate_current_schema(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validate and coerce object using current schema format.
    Supports three task types:
    - counseling_request: Suicidal ideation detection task
    - therapy_request: Therapy request classification task  
    - therapy_engagement: Therapy engagement detection task (safety_type optional)
    
    Args:
        obj: Parsed JSON object from model response
        
    Returns:
        Validated object with proper types, or None if invalid
    """
    try:
        # Determine which field variant is present
        if "therapy_engagement" in obj:
            request_field = "therapy_engagement"
            request_value = str(obj["therapy_engagement"]).strip()
            request_confidence_field = "therapy_engagement_confidence"
            valid_categories = THERAPY_ENGAGEMENT
            safety_optional = True  # safety_type not required for conversation-level classification
        elif "therapy_request" in obj:
            request_field = "therapy_request"
            request_value = str(obj["therapy_request"]).strip()
            request_confidence_field = "therapy_request_confidence"
            valid_categories = THERAPY_REQUESTS
            safety_optional = False
        elif "counseling_request" in obj:
            # Suicidal ideation detection task
            request_field = "counseling_request"
            request_value = str(obj["counseling_request"]).strip()
            request_confidence_field = "counseling_request_confidence"
            valid_categories = COUNSELING_REQUESTS
            safety_optional = False
        else:
            return None  # No valid field found
        
        # Validate request field
        if request_value not in valid_categories:
            return None
        
        # Handle safety_type - required unless safety_optional=True
        if safety_optional and "safety_type" not in obj:
            st = "no_SI"  # Default for conversation-level classification
            stc = 1.0
        else:
            st = str(obj["safety_type"]).strip()
            if st not in SAFETY_TYPES:
                return None
            stc = float(obj.get("safety_type_confidence", 1.0))
        
        crc = float(obj.get(request_confidence_field, 1.0))
        
        if not (0.0 <= stc <= 1.0 and 0.0 <= crc <= 1.0):
            return None
            
        # Return with normalized field names
        return {
            "safety_type": st,
            request_field: request_value, 
            "safety_type_confidence": stc,
            request_confidence_field: crc,
        }
    except (KeyError, ValueError, TypeError):
        return None


def validate_and_coerce(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validate and coerce model response object.
    
    Args:
        obj: Parsed JSON object from model response
        
    Returns:
        Validated and normalized object, or None if validation fails
    """
    if obj is None:
        return None
        
    return validate_current_schema(obj)


def create_empty_result() -> Dict[str, Any]:
    """Create an empty result object for failed parsing cases."""
    return {
        "safety_type": None,
        "safety_type_confidence": None,
        "therapy_request": None,
        "therapy_request_confidence": None,
        "therapy_engagement": None,
        "therapy_engagement_confidence": None,
    }