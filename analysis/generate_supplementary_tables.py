#!/usr/bin/env python3
"""
Generate Supplementary Tables S1-S3 from ground truth psychiatrist review data.

Table S1: SI detection synthetic data characterization
Table S2: Therapy request synthetic data characterization  
Table S3: Therapy engagement synthetic data characterization

Each table shows: Category, Description, Generated, Approved, Modified, Removed,
and Post_downsampling_n (counts in the finalized experiment inputs — same files as
`run_regulatory_simulation_paper_pipeline.py` / `utilities/confusion_matrix_audit.py`).

Ground truth (clinician review) sources:
- Table S1: data/inputs/intermediate_files/SI_psychiatrist_01_and_02_scores.csv
- Table S2: data/inputs/intermediate_files/therapy_request_psychiatrist_01_and_02_scores.csv
- Table S3: data/inputs/intermediate_files/therapy_engagement_psychiatrist_01_and_02_scores.csv

Finalized experiment inputs (Post_downsampling_n):
- SI: data/inputs/finalized_input_data/SI_finalized_sentences.csv
- Therapy request: data/inputs/finalized_input_data/therapy_request_finalized_sentences.csv
- Therapy engagement: data/inputs/finalized_input_data/therapy_engagement_finalized_sentences.csv
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "inputs" / "intermediate_files"
FINALIZED_DIR = ROOT / "data" / "inputs" / "finalized_input_data"
FINALIZED_SI_CSV = FINALIZED_DIR / "SI_finalized_sentences.csv"
FINALIZED_THERAPY_REQUEST_CSV = FINALIZED_DIR / "therapy_request_finalized_sentences.csv"
FINALIZED_THERAPY_ENGAGEMENT_CSV = FINALIZED_DIR / "therapy_engagement_finalized_sentences.csv"
# Psychiatrist review table uses active_si_abstract / preparatory_si; finalized SI files use
# experiment labels (prompts, config/constants.py).
SI_TABLE_INTERNAL_TO_FINALIZED_SAFETY_TYPE = {
    "active_si_abstract": "active_si_no_plan",
    "preparatory_si": "active_si_plan_with_intent_prep",
}
# Default when running this script standalone (pipeline passes --output-dir)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "Supplementary_Tables"


def write_supplementary_tables(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_table_s1().to_csv(output_dir / "table_s1_si_synthetic_data.csv", index=False)
    generate_table_s2().to_csv(output_dir / "table_s2_therapy_request_synthetic_data.csv", index=False)
    generate_table_s3().to_csv(output_dir / "table_s3_therapy_engagement_synthetic_data.csv", index=False)


def aggregate_te_subcategory(subcategory: str) -> str:
    if subcategory.startswith("SimulatedTherapy_"):
        parts = subcategory.replace("SimulatedTherapy_", "").split("_")
        if len(parts) >= 2:
            therapy_type = parts[1]
            return f"SimulatedTherapy_{therapy_type}"
    elif subcategory.startswith("Ambiguous_"):
        parts = subcategory.replace("Ambiguous_", "").split("_")
        if len(parts) >= 1:
            return f"Ambiguous_{parts[0]}"
    return subcategory


def _si_finalized_counts_by_table_category() -> tuple[dict[str, int], int]:
    edf = pd.read_csv(FINALIZED_SI_CSV)
    raw = edf.groupby("Safety type").size()
    leaf_keys = [
        "clearly_neutral",
        "clearly_happy",
        "clearly_sad",
        "clearly_angry",
        "ambiguous_emotional",
        "passive_si",
        "active_si_abstract",
        "active_si_plan_no_intent",
        "active_si_plan_with_intent",
        "preparatory_si",
    ]
    counts = {}
    for internal in leaf_keys:
        safety = SI_TABLE_INTERNAL_TO_FINALIZED_SAFETY_TYPE.get(internal, internal)
        counts[internal] = int(raw.get(safety, 0))
    return counts, len(edf)


def _therapy_request_finalized_counts_by_category() -> tuple[dict[str, int], int]:
    edf = pd.read_csv(FINALIZED_THERAPY_REQUEST_CSV)
    vc = edf.groupby("therapy_request").size()
    counts = {str(k): int(v) for k, v in vc.items()}
    return counts, len(edf)


def _therapy_engagement_finalized_counts_by_category() -> tuple[dict[str, int], int]:
    edf = pd.read_csv(FINALIZED_THERAPY_ENGAGEMENT_CSV)
    edf = edf.copy()
    edf["AggregatedSubCategory"] = edf["SubCategory"].apply(aggregate_te_subcategory)
    vc = edf.groupby("AggregatedSubCategory").size()
    counts = {str(k): int(v) for k, v in vc.items()}
    return counts, len(edf)


def generate_table_s1():
    """
    Generate Table S1: Characterizing SI detection synthetic data.
    
    Source: data/inputs/intermediate_files/SI_psychiatrist_01_and_02_scores.csv
    """
    df = pd.read_csv(DATA_DIR / "SI_psychiatrist_01_and_02_scores.csv")
    
    # Category definitions with descriptions
    categories = [
        # Non-SI categories
        ("clearly_neutral", "Neutral", "Factual or emotionally flat."),
        ("clearly_happy", "Happy", "Positive affect, gratitude, optimism."),
        ("clearly_sad", "Sad", "Grief, loneliness, disappointment."),
        ("clearly_angry", "Angry", "Frustration, irritability, hostility."),
        ("ambiguous_emotional", "Ambiguous Affect", "Vague or metaphorical phrasing that could be interpreted in multiple emotional ways and might confuse a model (not explicitly SI content)."),
        # Subtotal marker
        ("SUBTOTAL_NON_SI", "SUBTOTAL: Non-SI", ""),
        # SI categories
        ("passive_si", "SI: Passive", "Mild hopelessness, explicitly wishing not to exist or hoping for death without agency, but no active plan or intent."),
        ("active_si_abstract", "SI: Active (no plan)", "Thoughts of ending life or self-harm without specific plan or intent mentioned."),
        ("active_si_plan_no_intent", "SI: Active (plan, no intent)", "Mentions a specific, realistic method, with clear denial of desire or action. Plans must be specific."),
        ("active_si_plan_with_intent", "SI: Active (plan + intent)", "Expresses intent or decision for a specific, realistic method but no preparatory step mentioned. Plans must be specific."),
        ("preparatory_si", "SI: Preparatory", "Directly references having taken a specific preparatory action (e.g., obtained means, written note) for a specific, realistic method. Plans must be specific."),
        # Subtotal marker
        ("SUBTOTAL_SI", "SUBTOTAL: SI", ""),
        # Total marker
        ("TOTAL", "TOTAL", ""),
    ]
    
    non_si_types = ["clearly_neutral", "clearly_happy", "clearly_sad", "clearly_angry", "ambiguous_emotional"]
    si_types = ["passive_si", "active_si_abstract", "active_si_plan_no_intent", "active_si_plan_with_intent", "preparatory_si"]
    
    # Determine status for each statement based on P1 and P2 decisions
    def get_status(row):
        p1 = row['Psychiatrist_01']
        p2 = row['Psychiatrist_02']
        
        if p1 == 'KEPT_exact_match' and p2 == 'KEPT':
            return 'approved'
        elif p1 == 'KEPT_with_changes' and p2 == 'KEPT':
            return 'modified'
        else:
            return 'removed'
    
    df['status'] = df.apply(get_status, axis=1)
    eval_counts, eval_total = _si_finalized_counts_by_table_category()

    def post_n_sum(keys: list[str]) -> int:
        return sum(eval_counts[k] for k in keys)

    # Build table rows
    rows = []
    for internal_name, display_name, description in categories:
        if internal_name == "SUBTOTAL_NON_SI":
            subset = df[df['Safety type'].isin(non_si_types)]
            rows.append({
                'Category': display_name,
                'Description': description,
                'Generated': len(subset),
                'Approved': len(subset[subset['status'] == 'approved']),
                'Modified': len(subset[subset['status'] == 'modified']),
                'Removed': len(subset[subset['status'] == 'removed']),
                'Post_downsampling_n': post_n_sum(non_si_types),
            })
        elif internal_name == "SUBTOTAL_SI":
            subset = df[df['Safety type'].isin(si_types)]
            rows.append({
                'Category': display_name,
                'Description': description,
                'Generated': len(subset),
                'Approved': len(subset[subset['status'] == 'approved']),
                'Modified': len(subset[subset['status'] == 'modified']),
                'Removed': len(subset[subset['status'] == 'removed']),
                'Post_downsampling_n': post_n_sum(si_types),
            })
        elif internal_name == "TOTAL":
            rows.append({
                'Category': display_name,
                'Description': description,
                'Generated': len(df),
                'Approved': len(df[df['status'] == 'approved']),
                'Modified': len(df[df['status'] == 'modified']),
                'Removed': len(df[df['status'] == 'removed']),
                'Post_downsampling_n': eval_total,
            })
        else:
            subset = df[df['Safety type'] == internal_name]
            rows.append({
                'Category': display_name,
                'Description': description,
                'Generated': len(subset),
                'Approved': len(subset[subset['status'] == 'approved']),
                'Modified': len(subset[subset['status'] == 'modified']),
                'Removed': len(subset[subset['status'] == 'removed']),
                'Post_downsampling_n': eval_counts[internal_name],
            })

    assert rows[-1]['Post_downsampling_n'] == eval_total
    return pd.DataFrame(rows)


def generate_table_s2():
    """
    Generate Table S2: Characterizing therapy request synthetic data.
    
    Source: data/inputs/intermediate_files/therapy_request_psychiatrist_01_and_02_scores.csv
    """
    df = pd.read_csv(DATA_DIR / "therapy_request_psychiatrist_01_and_02_scores.csv")
    
    # Determine status based on P1 and P2 decisions
    def get_status(row):
        p1 = row['Psychiatrist_01']
        p2 = row['Psychiatrist_02']
        
        if p1 == 'REMOVED' or p2 == 'REMOVED':
            return 'removed'
        elif p1 == 'KEPT_with_changes':
            return 'modified'
        else:
            return 'approved'
    
    df['status'] = df.apply(get_status, axis=1)
    eval_counts, eval_total = _therapy_request_finalized_counts_by_category()

    def post_n_sum(keys: list[str]) -> int:
        return sum(eval_counts[k] for k in keys)

    # Category definitions with descriptions
    categories = [
        # Declarative Statements
        ("Neutral Declarative Statements", "Declarative: Neutral", "Factual statements without emotional content."),
        ("Affect-Containing Declarative Statements - Clearly Happy", "Declarative: Happy", "Positive affect declarative statements."),
        ("Affect-Containing Declarative Statements - Clearly Sad", "Declarative: Sad", "Sad affect declarative statements."),
        ("Affect-Containing Declarative Statements - Clearly Angry", "Declarative: Angry", "Angry affect declarative statements."),
        ("SUBTOTAL_DECLARATIVE", "SUBTOTAL: Declarative", ""),
        # Non-Therapeutic Questions
        ("Neutral Non-Therapeutic Questions", "Non-Therapeutic Question: Neutral", "Neutral questions not seeking therapy."),
        ("Affect-Containing Non-Therapeutic Questions - Clearly Happy", "Non-Therapeutic Question: Happy", "Happy affect non-therapeutic questions."),
        ("Affect-Containing Non-Therapeutic Questions - Clearly Sad", "Non-Therapeutic Question: Sad", "Sad affect non-therapeutic questions."),
        ("Affect-Containing Non-Therapeutic Questions - Clearly Angry", "Non-Therapeutic Question: Angry", "Angry affect non-therapeutic questions."),
        ("SUBTOTAL_NON_THERAPEUTIC", "SUBTOTAL: Non-Therapeutic Questions", ""),
        # Explicit Therapy Requests
        ("Neutral Explicit Requests for Therapy", "Explicit Therapy Request: Neutral", "Direct therapy requests without emotional content."),
        ("Affect-Containing Explicit Requests for Therapy - Clearly Happy", "Explicit Therapy Request: Happy", "Happy affect therapy requests."),
        ("Affect-Containing Explicit Requests for Therapy - Clearly Sad", "Explicit Therapy Request: Sad", "Sad affect therapy requests."),
        ("Affect-Containing Explicit Requests for Therapy - Clearly Angry", "Explicit Therapy Request: Angry", "Angry affect therapy requests."),
        ("SUBTOTAL_EXPLICIT", "SUBTOTAL: Explicit Therapy Requests", ""),
        ("TOTAL", "TOTAL", ""),
    ]
    
    declarative_types = [
        "Neutral Declarative Statements",
        "Affect-Containing Declarative Statements - Clearly Happy",
        "Affect-Containing Declarative Statements - Clearly Sad",
        "Affect-Containing Declarative Statements - Clearly Angry"
    ]
    non_therapeutic_types = [
        "Neutral Non-Therapeutic Questions",
        "Affect-Containing Non-Therapeutic Questions - Clearly Happy",
        "Affect-Containing Non-Therapeutic Questions - Clearly Sad",
        "Affect-Containing Non-Therapeutic Questions - Clearly Angry"
    ]
    explicit_types = [
        "Neutral Explicit Requests for Therapy",
        "Affect-Containing Explicit Requests for Therapy - Clearly Happy",
        "Affect-Containing Explicit Requests for Therapy - Clearly Sad",
        "Affect-Containing Explicit Requests for Therapy - Clearly Angry"
    ]
    
    # Build table rows
    rows = []
    for internal_name, display_name, description in categories:
        if internal_name == "SUBTOTAL_DECLARATIVE":
            subset = df[df['Counseling Request'].isin(declarative_types)]
        elif internal_name == "SUBTOTAL_NON_THERAPEUTIC":
            subset = df[df['Counseling Request'].isin(non_therapeutic_types)]
        elif internal_name == "SUBTOTAL_EXPLICIT":
            subset = df[df['Counseling Request'].isin(explicit_types)]
        elif internal_name == "TOTAL":
            subset = df
        else:
            subset = df[df['Counseling Request'] == internal_name]
        
        if internal_name == "SUBTOTAL_DECLARATIVE":
            post_n = post_n_sum(declarative_types)
        elif internal_name == "SUBTOTAL_NON_THERAPEUTIC":
            post_n = post_n_sum(non_therapeutic_types)
        elif internal_name == "SUBTOTAL_EXPLICIT":
            post_n = post_n_sum(explicit_types)
        elif internal_name == "TOTAL":
            post_n = eval_total
        else:
            post_n = eval_counts[internal_name]

        rows.append({
            'Category': display_name,
            'Description': description,
            'Generated': len(subset),
            'Approved': len(subset[subset['status'] == 'approved']),
            'Modified': len(subset[subset['status'] == 'modified']),
            'Removed': len(subset[subset['status'] == 'removed']),
            'Post_downsampling_n': post_n,
        })

    assert rows[-1]['Post_downsampling_n'] == eval_total
    return pd.DataFrame(rows)


def generate_table_s3():
    """
    Generate Table S3: Characterizing therapy engagement synthetic data.
    
    Source: data/inputs/intermediate_files/therapy_engagement_psychiatrist_01_and_02_scores.csv
    """
    df = pd.read_csv(DATA_DIR / "therapy_engagement_psychiatrist_01_and_02_scores.csv")
    
    # Group by conversation (Example_ID) - take first row per conversation
    conversations = df.groupby('Example_ID').first().reset_index()
    conversations['AggregatedSubCategory'] = conversations['SubCategory'].apply(aggregate_te_subcategory)
    
    # Determine status based on P1 and P2 decisions
    def get_status(row):
        p1 = row['Psychiatrist_01']
        p2 = row['Psychiatrist_02']
        
        if p1 == 'REMOVED' or p2 == 'REMOVED':
            return 'removed'
        elif p1 == 'KEPT_with_changes':
            return 'modified'
        else:
            return 'approved'
    
    conversations['status'] = conversations.apply(get_status, axis=1)
    eval_counts, eval_total = _therapy_engagement_finalized_counts_by_category()

    def post_n_sum(keys: list[str]) -> int:
        return sum(eval_counts[k] for k in keys)

    # Category definitions with descriptions
    categories = [
        # Non-Therapeutic
        ("NonTherapeutic_CreativeWriting", "Non-Therapeutic: Creative Writing", "Creative writing assistance, storytelling."),
        ("NonTherapeutic_InfoSeeking", "Non-Therapeutic: Info Seeking", "General information queries."),
        ("NonTherapeutic_PlanningOrg", "Non-Therapeutic: Planning/Organization", "Scheduling, planning assistance."),
        ("NonTherapeutic_PracticalTask", "Non-Therapeutic: Practical Task", "Practical task assistance."),
        ("NonTherapeutic_TechnicalCoding", "Non-Therapeutic: Technical/Coding", "Technical or coding assistance."),
        ("SUBTOTAL_NON_THERAPEUTIC", "SUBTOTAL: Non-Therapeutic", ""),
        # Ambiguous
        ("Ambiguous_DisclosureBoundary", "Ambiguous: Detected Disclosure", "User disclosure at conversation boundary."),
        ("Ambiguous_InfoPathology", "Ambiguous: Info - Pathology", "Information about mental health conditions."),
        ("Ambiguous_InfoTherapy", "Ambiguous: Info - Therapy", "Information about therapy approaches."),
        ("SUBTOTAL_AMBIGUOUS", "SUBTOTAL: Ambiguous", ""),
        # Therapeutic
        ("SimulatedTherapy_CognitiveTechniqueConcept", "Therapeutic: Cognitive Technique", "Cognitive restructuring techniques."),
        ("SimulatedTherapy_SkillConcept", "Therapeutic: CBT/DBT Skill", "CBT or DBT skill instruction."),
        ("SimulatedTherapy_PsychoanalyticConcept", "Therapeutic: Psychodynamic", "Psychodynamic interpretation or exploration."),
        ("SimulatedTherapy_DiagnosisSuggestion", "Therapeutic: Diagnosis", "Diagnostic clarification or suggestion."),
        ("SimulatedTherapy_MedicationMention", "Therapeutic: Medication", "Medication discussion or recommendation."),
        ("SUBTOTAL_THERAPEUTIC", "SUBTOTAL: Therapeutic", ""),
        ("TOTAL", "TOTAL", ""),
    ]
    
    non_therapeutic_types = [
        "NonTherapeutic_CreativeWriting", "NonTherapeutic_InfoSeeking", 
        "NonTherapeutic_PlanningOrg", "NonTherapeutic_PracticalTask", "NonTherapeutic_TechnicalCoding"
    ]
    ambiguous_types = ["Ambiguous_DisclosureBoundary", "Ambiguous_InfoPathology", "Ambiguous_InfoTherapy"]
    therapeutic_types = [
        "SimulatedTherapy_CognitiveTechniqueConcept", "SimulatedTherapy_SkillConcept",
        "SimulatedTherapy_PsychoanalyticConcept", "SimulatedTherapy_DiagnosisSuggestion",
        "SimulatedTherapy_MedicationMention"
    ]
    
    # Build table rows
    rows = []
    for internal_name, display_name, description in categories:
        if internal_name == "SUBTOTAL_NON_THERAPEUTIC":
            subset = conversations[conversations['AggregatedSubCategory'].isin(non_therapeutic_types)]
        elif internal_name == "SUBTOTAL_AMBIGUOUS":
            subset = conversations[conversations['AggregatedSubCategory'].isin(ambiguous_types)]
        elif internal_name == "SUBTOTAL_THERAPEUTIC":
            subset = conversations[conversations['AggregatedSubCategory'].isin(therapeutic_types)]
        elif internal_name == "TOTAL":
            subset = conversations
        else:
            subset = conversations[conversations['AggregatedSubCategory'] == internal_name]

        if internal_name == "SUBTOTAL_NON_THERAPEUTIC":
            post_n = post_n_sum(non_therapeutic_types)
        elif internal_name == "SUBTOTAL_AMBIGUOUS":
            post_n = post_n_sum(ambiguous_types)
        elif internal_name == "SUBTOTAL_THERAPEUTIC":
            post_n = post_n_sum(therapeutic_types)
        elif internal_name == "TOTAL":
            post_n = eval_total
        else:
            post_n = eval_counts[internal_name]

        rows.append({
            'Category': display_name,
            'Description': description,
            'Generated': len(subset),
            'Approved': len(subset[subset['status'] == 'approved']),
            'Modified': len(subset[subset['status'] == 'modified']),
            'Removed': len(subset[subset['status'] == 'removed']),
            'Post_downsampling_n': post_n,
        })

    assert rows[-1]['Post_downsampling_n'] == eval_total
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate Supplementary Tables S1-S3.")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV output (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = ap.parse_args()
    write_supplementary_tables(args.output_dir)
    print(f"Wrote S1-S3 CSVs to {args.output_dir.resolve()}")
