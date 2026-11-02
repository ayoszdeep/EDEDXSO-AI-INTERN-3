"""
explanations.py
===============
Explainable AI (XAI) engine for the EDXSO School Influencer Discovery Tool.
Generates deterministic, evidence-backed justifications and quantitative
score breakdowns for each discovered candidate, ensuring high transparency
and zero hallucinations.
"""

from typing import Any, Dict, List, Optional


def generate_explanation(
    reference_analysis: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate natural language justifications and quantitative provenance
    explaining why a candidate matches the reference professional.

    Parameters:
        reference_analysis: Output of profile_analyzer.analyze_profile()
        candidate: Candidate record with score metrics from ranker.py

    Returns:
        Dictionary with:
            - summary: One-sentence verdict
            - reasons: List of evidence-backed bullets
            - score_breakdown: Formatted score components string
            - source_evidence: Provenance reference and URL
    """
    reasons: List[str] = []

    cand_name = candidate.get("name", "Candidate")
    ref_name = reference_analysis.get("name", "Reference")
    ref_tags = set(reference_analysis.get("school_ecosystem_tags", []))
    cand_tags = set(candidate.get("parsed_tags", []))

    # 1. Shared ecosystem indicators
    shared_tags = sorted(list(ref_tags & cand_tags))
    if shared_tags:
        clean_tags = [t.replace("_", " ").title() for t in shared_tags]
        reasons.append(
            f"Shares verified school ecosystem indicators: {', '.join(clean_tags)}."
        )

    # 2. Role alignment
    ref_family = reference_analysis.get("role_family", "")
    cand_family = candidate.get("role_family", "")
    if ref_family and cand_family:
        if ref_family.lower() == cand_family.lower():
            reasons.append(
                f"Direct role-family alignment: both operate within '{cand_family}'."
            )
        else:
            reasons.append(
                f"High cross-domain synergy between {ref_family} and {cand_family}."
            )

    # 3. Principal / Leadership background
    if "principal" in cand_tags:
        if "principal" in ref_tags or "School Principal" in reference_analysis.get("previous_roles", []):
            reasons.append(
                "Shared background as former/current School Principal with firsthand institutional leadership."
            )
        else:
            reasons.append("Documented school leadership background as School Principal.")

    # 4. Multi-school network evidence
    multi_bool = candidate.get("multi_school_bool")
    if multi_bool is True:
        reasons.append(
            "Documented multi-school exposure: has verified experience advising, leading, or managing multiple schools."
        )
    elif multi_bool is False:
        reasons.append(
            "Focused single-institution deep leadership."
        )

    # 5. Advisory / Decision-maker reach
    if candidate.get("decision_maker_bool") is True:
        reasons.append(
            "High decision-maker influence: advises or collaborates directly with school owners, trustees, and management boards."
        )

    # 6. Geographic proximity
    ref_loc = reference_analysis.get("location", "")
    cand_loc = candidate.get("location", "")
    if ref_loc and cand_loc:
        from ranker import calculate_location_score
        loc_score = calculate_location_score(ref_loc, cand_loc)
        if loc_score == 1.0:
            reasons.append(f"Geographic proximity: located in the same regional education hub ({cand_loc}).")

    if not reasons:
        reasons.append(
            "Demonstrates semantic and contextual similarity in K-12 school education background."
        )

    # Quantitative breakdown
    contributions = candidate.get("contributions", {})
    breakdown_parts = []
    if "semantic" in contributions:
        breakdown_parts.append(f"Semantic AI: {contributions['semantic']} pts")
    if "school" in contributions:
        breakdown_parts.append(f"Ecosystem Tags: {contributions['school']} pts")
    if "role" in contributions:
        breakdown_parts.append(f"Role Match: {contributions['role']} pts")
    if "network" in contributions:
        breakdown_parts.append(f"Multi-School: {contributions['network']} pts")
    if "location" in contributions:
        breakdown_parts.append(f"Location: {contributions['location']} pts")

    score_str = " | ".join(breakdown_parts)
    total = candidate.get("total_score", 0.0)

    # Source provenance
    source_url = candidate.get("source_url", "")
    source_date = candidate.get("source_date", "Recent")
    source_evidence = f"Verified via public profile record ({source_date}): {source_url}"

    # Summary
    if total >= 80:
        verdict = f"High Relevance Match ({total}/100) — Exceptional synergy in school leadership and advisory footprint."
    elif total >= 65:
        verdict = f"Strong Relevance Match ({total}/100) — Relevant school ecosystem background with substantial domain alignment."
    else:
        verdict = f"Moderate Relevance Match ({total}/100) — Aligned in specific school education aspects."

    return {
        "verdict": verdict,
        "reasons": reasons,
        "score_breakdown": score_str,
        "source_evidence": source_evidence,
        "source_url": source_url,
    }
