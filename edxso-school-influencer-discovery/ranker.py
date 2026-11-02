"""
ranker.py
=========
Intelligent hybrid ranking engine for the EDXSO School Influencer Discovery Tool.
Combines:
  1. AI Semantic Similarity (45% default)
  2. School Ecosystem Relevance (20% default)
  3. Role & Experience Similarity (15% default)
  4. Multi-School Network Exposure (15% default)
  5. Location Relevance (5% default)
Supports dynamic weight tuning, graceful missing-data renormalization,
candidate data completeness scoring, and strict reference exclusion.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from logger_config import setup_logger

logger = setup_logger("ranker")

DEFAULT_WEIGHTS = {
    "semantic": 0.45,
    "school": 0.20,
    "role": 0.15,
    "network": 0.15,
    "location": 0.05,
}

WEIGHTS = DEFAULT_WEIGHTS.copy()

METRO_CLUSTERS = {
    "delhi ncr": {"delhi", "new delhi", "noida", "greater noida", "gurgaon", "gurugram", "faridabad", "ghaziabad"},
    "mumbai": {"mumbai", "navi mumbai", "thane"},
    "bengaluru": {"bengaluru", "bangalore"},
    "hyderabad": {"hyderabad", "secunderabad"},
    "kolkata": {"kolkata", "calcutta"},
    "chennai": {"chennai", "madras"},
    "pune": {"pune"},
}

ROLE_COMPATIBILITY = {
    ("Education Consulting", "School Leadership"): 0.85,
    ("School Leadership", "Education Consulting"): 0.85,
    ("Education Consulting", "School Network Management"): 0.85,
    ("School Network Management", "Education Consulting"): 0.85,
    ("School Leadership", "School Network Management"): 0.90,
    ("School Network Management", "School Leadership"): 0.90,
    ("School Governance", "School Network Management"): 0.85,
    ("School Network Management", "School Governance"): 0.85,
    ("School Governance", "Education Consulting"): 0.75,
    ("Education Consulting", "School Governance"): 0.75,
    ("Education Entrepreneurship", "Education Consulting"): 0.75,
    ("Education Consulting", "Education Entrepreneurship"): 0.75,
    ("Education Entrepreneurship", "School Governance"): 0.70,
    ("School Governance", "Education Entrepreneurship"): 0.70,
    ("School Leadership", "School Governance"): 0.80,
    ("School Governance", "School Leadership"): 0.80,
}


def tag_similarity(reference_tags: List[str], candidate_tags: List[str]) -> float:
    """
    Compute Jaccard similarity between reference and candidate tags:
    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    set_ref = set(t.strip().lower() for t in reference_tags if t.strip())
    set_cand = set(t.strip().lower() for t in candidate_tags if t.strip())

    union = set_ref | set_cand
    if not union:
        return 0.0
    intersection = set_ref & set_cand
    return len(intersection) / len(union)


def calculate_role_similarity(
    ref_role_family: str,
    cand_role_family: str,
    ref_exp: Optional[int] = None,
    cand_exp: Optional[int] = None,
) -> float:
    """
    Calculate similarity between professional role families with experience adjustment.
    """
    if not ref_role_family or not cand_role_family:
        return 0.5

    if ref_role_family.lower() == cand_role_family.lower():
        base = 1.0
    else:
        pair = (ref_role_family, cand_role_family)
        base = ROLE_COMPATIBILITY.get(pair, 0.4)

    # Experience proximity bonus if both verified
    if ref_exp is not None and cand_exp is not None:
        delta = abs(ref_exp - cand_exp)
        if delta <= 3:
            base = min(1.0, base + 0.05)
        elif delta <= 7:
            base = min(1.0, base + 0.02)

    return float(base)


def calculate_network_score(
    candidate_record: Dict[str, Any],
) -> Optional[float]:
    """
    Calculate multi-school network exposure score.
    Returns None if verified evidence is unavailable (allowing renormalization).
    """
    multi_bool = candidate_record.get("multi_school_bool")
    bio = str(candidate_record.get("biography", "")).lower()
    exp = str(candidate_record.get("school_experience", "")).lower()
    combined = f"{bio} {exp}"

    # Explicit multi-school boolean
    if multi_bool is True:
        if re.search(r"\b(\d{1,2}\+?\s*schools|network of|group of schools|multiple campuses|chain)\b", combined):
            return 1.0
        return 0.9

    if multi_bool is False:
        if re.search(r"\b(principal|headmaster)\b", combined):
            return 0.5
        return 0.2

    if re.search(r"\b(multiple schools|school network|several schools|cluster|pan-india)\b", combined):
        return 0.85

    return None


def _get_metro(city_str: str) -> Optional[str]:
    c = city_str.lower().strip()
    for metro, cities in METRO_CLUSTERS.items():
        if any(cit in c for cit in cities):
            return metro
    return None


def calculate_location_score(ref_location: str, cand_location: str) -> Optional[float]:
    """
    Calculate geographic relevance score.
    Returns None if either location is missing/unknown.
    """
    if not ref_location or not cand_location:
        return None

    r_clean = ref_location.strip().lower()
    c_clean = cand_location.strip().lower()

    if not r_clean or not c_clean:
        return None

    if r_clean == c_clean or r_clean in c_clean or c_clean in r_clean:
        return 1.0

    ref_metro = _get_metro(r_clean)
    cand_metro = _get_metro(c_clean)
    if ref_metro and cand_metro and ref_metro == cand_metro:
        return 1.0

    is_ref_india = any(w in r_clean for w in ["india", "delhi", "mumbai", "bengaluru", "bangalore", "hyderabad", "pune", "kolkata", "chennai", "noida", "gurgaon"])
    is_cand_india = any(w in c_clean for w in ["india", "delhi", "mumbai", "bengaluru", "bangalore", "hyderabad", "pune", "kolkata", "chennai", "noida", "gurgaon"])

    if is_ref_india and is_cand_india:
        return 0.5

    return 0.2


def calculate_final_score(
    components: Dict[str, Optional[float]],
    weights_dict: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute total normalized score (0-100) using available components and renormalization.
    """
    active_weights = weights_dict or WEIGHTS
    available = [
        (key, active_weights[key], val)
        for key, val in components.items()
        if key in active_weights and val is not None
    ]

    if not available:
        return 0.0, {}

    total_weight = sum(w for _, w, _ in available)
    if total_weight <= 0:
        return 0.0, {}

    contributions = {}
    weighted_sum = 0.0

    for key, weight, val in available:
        normalized_weight = weight / total_weight
        component_points = val * normalized_weight * 100.0
        contributions[key] = round(component_points, 1)
        weighted_sum += component_points

    total_score = round(min(100.0, max(0.0, weighted_sum)), 1)
    return total_score, contributions


def compute_candidate_completeness(cand: Dict[str, Any]) -> float:
    """Compute data verification completeness percentage for a candidate."""
    checks = [
        bool(cand.get("name")),
        bool(cand.get("current_role")),
        bool(cand.get("organisation")),
        bool(cand.get("location")),
        bool(cand.get("biography")),
        cand.get("experience_years") is not None and not pd.isna(cand.get("experience_years")),
        cand.get("multi_school_bool") is not None,
        cand.get("decision_maker_bool") is not None,
        bool(cand.get("source_url")),
        bool(cand.get("source_date")),
    ]
    return round((sum(checks) / len(checks)) * 100.0, 1)


def rank_candidates(
    reference_analysis: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    semantic_scores: np.ndarray,
    min_score: float = 0.0,
    custom_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Score and rank candidate profiles against reference profile analysis.
    """
    start_time = time.time()
    ref_name = str(reference_analysis.get("name", "")).strip().lower()
    ref_li = str(reference_analysis.get("linkedin_url", "")).strip().lower()
    ref_tags = reference_analysis.get("school_ecosystem_tags", [])
    ref_role_family = reference_analysis.get("role_family", "")
    ref_exp = reference_analysis.get("experience_years")
    ref_loc = reference_analysis.get("location", "")

    effective_weights = custom_weights or WEIGHTS
    ranked_results = []
    excluded_count = 0
    renormalized_count = 0

    for idx, cand in enumerate(candidates):
        cand_name = str(cand.get("name", "")).strip().lower()
        cand_li = str(cand.get("linkedin_url", "")).strip().lower()

        # Strict exclusion of reference professional
        if ref_name and cand_name and (ref_name == cand_name):
            excluded_count += 1
            logger.info("Excluded reference person '%s' from discovery results", cand.get("name"))
            continue
        if ref_li and cand_li and (ref_li == cand_li):
            excluded_count += 1
            logger.info("Excluded reference person by LinkedIn URL from discovery results")
            continue

        sem_score = float(semantic_scores[idx]) if idx < len(semantic_scores) else 0.0
        school_score = tag_similarity(ref_tags, cand.get("parsed_tags", []))

        cand_family = cand.get("role_family")
        if not cand_family:
            from profile_analyzer import extract_role_family
            cand_family = extract_role_family(cand.get("current_role", ""), cand.get("biography", ""))

        cand_exp = cand.get("experience_years")
        if pd.isna(cand_exp):
            cand_exp = None
        else:
            try:
                cand_exp = int(cand_exp)
            except (ValueError, TypeError):
                cand_exp = None

        role_score = calculate_role_similarity(ref_role_family, cand_family, ref_exp, cand_exp)
        network_score = calculate_network_score(cand)
        location_score = calculate_location_score(ref_loc, cand.get("location", ""))

        components = {
            "semantic": sem_score,
            "school": school_score,
            "role": role_score,
            "network": network_score,
            "location": location_score,
        }

        # Track renormalization
        if network_score is None or location_score is None:
            renormalized_count += 1

        total_score, contributions = calculate_final_score(components, effective_weights)

        if total_score < min_score:
            continue

        completeness = compute_candidate_completeness(cand)

        result_item = dict(cand)
        result_item.update({
            "total_score": total_score,
            "role_family": cand_family,
            "raw_semantic": round(sem_score, 3),
            "raw_school": round(school_score, 3),
            "raw_role": round(role_score, 3),
            "raw_network": round(network_score, 3) if network_score is not None else None,
            "raw_location": round(location_score, 3) if location_score is not None else None,
            "contributions": contributions,
            "components": components,
            "data_completeness": completeness,
            "confidence_tier": "High Confidence" if completeness >= 80 else "Moderate Confidence",
        })
        ranked_results.append(result_item)

    ranked_results.sort(key=lambda x: x["total_score"], reverse=True)
    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    logger.info(
        "Ranked %d candidates in %.2f ms. Top score: %.1f, Excluded: %d, Renormalized: %d",
        len(ranked_results),
        elapsed_ms,
        ranked_results[0]["total_score"] if ranked_results else 0.0,
        excluded_count,
        renormalized_count,
    )

    return ranked_results
