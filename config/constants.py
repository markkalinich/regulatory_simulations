#!/usr/bin/env python3
"""
System Constants - Core constants used throughout the system.

This module contains the core constants that were originally in config.py,
moved here for better organization within the config package.
"""

# Safety types for current system
SAFETY_TYPES = [
    # Non-suicidal categories
    "clearly_happy",
    "clearly_neutral", 
    "clearly_sad",
    "clearly_angry",
    "ambiguous_emotional",
    "no_SI",  # Therapy request classification (non-SI only)
    # Suicidal ideation categories
    "passive_si",
    "active_si_no_plan",
    "active_si_plan_no_intent",
    "active_si_plan_with_intent",
    "active_si_plan_with_intent_prep",
]

# Therapy request categories (v3 - new format)
THERAPY_REQUESTS = [
    "declarative",
    "non_therapeutic_question",
    "explicit_therapy_request",
]

# Therapy delivery categories (v3)
THERAPY_DELIVERY = [
    "informational_support",
    "empathic_acknowledgment",
    "therapeutic_engagement",
]

# Therapy engagement categories (v2 - conversation-level classification)
THERAPY_ENGAGEMENT = [
    "non_therapeutic",
    "ambiguous_engagement",
    "simulated_therapy",
]

# SI categories for binary classification (matches model output taxonomy)
# NOTE: ambiguous_emotional is NOT SI per psychiatrist review
SI_POSITIVE_CATEGORIES = [
    "passive_si",
    "active_si_no_plan",
    "active_si_plan_no_intent",
    "active_si_plan_with_intent",
    "active_si_plan_with_intent_prep"
]

# Model-specific prompt modifications
# Some models require special prompt suffixes to control their behavior
MODEL_SPECIFIC_PROMPT_SUFFIXES = {
    'qwen': '/no_think',  # Disables chain-of-thought reasoning in Qwen models
                          # Without this, Qwen outputs verbose reasoning before JSON response
                          # which breaks parsing. See: https://qwenlm.github.io/blog/qwen2.5/
}

# ============================================================================
# DISPLAY LABELS FOR PLOTS
# Human-readable labels for categories (used in visualizations/manuscripts)
# ============================================================================

# SI category display labels
SI_LABELS = {
    # Emotions
    'clearly_happy': 'Happy',
    'clearly_neutral': 'Neutral',
    'clearly_sad': 'Sad',
    'clearly_angry': 'Angry',
    'ambiguous_emotional': 'Ambiguous',
    # SI categories
    'passive_si': 'Passive SI',
    'active_si_no_plan': 'Active SI\n(No Plan)',
    'active_si_plan_no_intent': 'Active SI\n(Plan, No Intent)',
    'active_si_plan_with_intent': 'Active SI\n(Plan & Intent)',
    'active_si_plan_with_intent_prep': 'Active SI\n(Plan, Intent\n& Prep)'
}

# Therapy request display labels
THERAPY_REQUEST_LABELS = {
    'Neutral Declarative Statements': 'Neutral\nStatement',
    'Neutral Non-Therapeutic Questions': 'Neutral\nQuestion',
    'Neutral Explicit Requests for Therapy': 'Neutral\nRequest',
    'Affect-Containing Declarative Statements - Clearly Happy': 'Happy\nStatement',
    'Affect-Containing Non-Therapeutic Questions - Clearly Happy': 'Happy\nQuestion',
    'Affect-Containing Explicit Requests for Therapy - Clearly Happy': 'Happy\nRequest',
    'Affect-Containing Declarative Statements - Clearly Sad': 'Sad\nStatement',
    'Affect-Containing Non-Therapeutic Questions - Clearly Sad': 'Sad\nQuestion',
    'Affect-Containing Explicit Requests for Therapy - Clearly Sad': 'Sad\nRequest',
    'Affect-Containing Declarative Statements - Clearly Angry': 'Angry\nStatement',
    'Affect-Containing Non-Therapeutic Questions - Clearly Angry': 'Angry\nQuestion',
    'Affect-Containing Explicit Requests for Therapy - Clearly Angry': 'Angry\nRequest'
}

# Therapy request category ordering (for confusion matrices, plots)
# Grouped by: declarative statements, then questions, then explicit requests
THERAPY_REQUEST_CATEGORY_ORDER = [
    # Declarative statements
    'Neutral Declarative Statements',
    'Affect-Containing Declarative Statements - Clearly Happy',
    'Affect-Containing Declarative Statements - Clearly Sad',
    'Affect-Containing Declarative Statements - Clearly Angry',
    # Non-therapeutic questions
    'Neutral Non-Therapeutic Questions',
    'Affect-Containing Non-Therapeutic Questions - Clearly Happy',
    'Affect-Containing Non-Therapeutic Questions - Clearly Sad',
    'Affect-Containing Non-Therapeutic Questions - Clearly Angry',
    # Explicit therapy requests
    'Neutral Explicit Requests for Therapy',
    'Affect-Containing Explicit Requests for Therapy - Clearly Happy',
    'Affect-Containing Explicit Requests for Therapy - Clearly Sad',
    'Affect-Containing Explicit Requests for Therapy - Clearly Angry'
]

# Therapy engagement display labels - helper function for dynamic categories
def get_therapy_engagement_label(subcategory: str) -> str:
    """Get display label for therapy engagement subcategory."""
    if 'NonTherapeutic' in subcategory:
        label = subcategory.replace('NonTherapeutic_', '')
        names = {
            'PlanningOrganizing': 'Planning/\nOrganizing',
            'PracticalTask': 'Practical\nTask',
            'TechnicalCoding': 'Technical/\nCoding'
        }
        return names.get(label, label.replace('_', '\n'))
    elif 'AmbiguousEngagement' in subcategory:
        label = subcategory.replace('AmbiguousEngagement_', '')
        names = {
            'HealthInfo': 'Health\nInfo',
            'EmotionalSupport': 'Emotional\nSupport'
        }
        return names.get(label, label.replace('_', '\n'))
    elif 'SimulatedTherapy' in subcategory:
        label = subcategory.replace('SimulatedTherapy_', '')
        names = {
            'DiagnosisSuggestion': 'Diagnosis',
            'MedicationMention': 'Med\nRecommendation',
            'PsychoanalyticConcept': 'Psychodynamic',
            'SkillConcept': 'CBT/DBT-based\nSkill',
            'CognitiveTechniqueConcept': 'Cognitive\nTechnique'
        }
        return names.get(label, label.replace('_', '\n'))
    return subcategory.replace('_', '\n')
