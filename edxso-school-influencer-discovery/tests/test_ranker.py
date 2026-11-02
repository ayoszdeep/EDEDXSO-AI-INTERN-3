"""
test_ranker.py
==============
Unit tests for ranking calculation, Jaccard similarity, role compatibility,
network exposure, and renormalization.
"""

import numpy as np
import pytest
from ranker import (
    tag_similarity,
    calculate_role_similarity,
    calculate_network_score,
    calculate_location_score,
    calculate_final_score,
    rank_candidates,
    WEIGHTS,
)


def test_tag_similarity_jaccard():
    # Identical tags
    assert tag_similarity(["principal", "consultant"], ["principal", "consultant"]) == 1.0
    # Disjoint tags
    assert tag_similarity(["principal"], ["software_engineer"]) == 0.0
    # Overlapping: union = 3, intersection = 1 -> 1/3 = 0.333...
    sim = tag_similarity(["principal", "consultant"], ["consultant", "director"])
    assert round(sim, 2) == 0.33
    # Empty tags
    assert tag_similarity([], []) == 0.0


def test_role_similarity():
    # Exact match
    assert calculate_role_similarity("School Leadership", "School Leadership") == 1.0
    # Known adjacent transition (Leadership <-> Consulting)
    adj_score = calculate_role_similarity("School Leadership", "Education Consulting")
    assert adj_score >= 0.8
    # Unrelated
    unrel = calculate_role_similarity("School Leadership", "Software Development")
    assert unrel <= 0.5


def test_network_score():
    # Verified multi-school
    cand_yes = {"multi_school_bool": True, "biography": "Advises 10+ schools across Delhi."}
    assert calculate_network_score(cand_yes) == 1.0

    # Single school
    cand_no = {"multi_school_bool": False, "biography": "Principal of single school."}
    assert calculate_network_score(cand_no) == 0.5

    # Completely unknown -> should return None for renormalization
    cand_unknown = {"multi_school_bool": None, "biography": "Educator background."}
    assert calculate_network_score(cand_unknown) is None


def test_location_score():
    # Exact city match
    assert calculate_location_score("Delhi NCR", "Delhi NCR") == 1.0
    # Metro cluster match (Delhi <-> Noida)
    assert calculate_location_score("Delhi NCR", "Noida, Delhi NCR") == 1.0
    # Both in India
    assert calculate_location_score("Delhi NCR", "Bengaluru, India") == 0.5
    # Missing location -> None (renormalized)
    assert calculate_location_score("Delhi NCR", "") is None


def test_renormalization_without_missing():
    # When all components present, sum should strictly follow weights
    components = {
        "semantic": 1.0,
        "school": 1.0,
        "role": 1.0,
        "network": 1.0,
        "location": 1.0,
    }
    score, contribs = calculate_final_score(components)
    assert score == 100.0
    assert contribs["semantic"] == 45.0
    assert contribs["school"] == 20.0
    assert contribs["role"] == 15.0
    assert contribs["network"] == 15.0
    assert contribs["location"] == 5.0


def test_renormalization_with_missing_component():
    # If network and location are None, weights renormalize across semantic, school, role (0.45, 0.20, 0.15 = 0.80)
    components = {
        "semantic": 1.0,
        "school": 1.0,
        "role": 1.0,
        "network": None,
        "location": None,
    }
    score, contribs = calculate_final_score(components)
    assert score == 100.0  # 1.0 on all available components should equal 100.0


def test_reference_person_exclusion():
    ref_analysis = {
        "name": "Dr. Amit Sharma",
        "linkedin_url": "https://linkedin.com/in/amit-sharma-ref",
        "role_family": "Education Consulting",
        "school_ecosystem_tags": ["principal", "consultant"],
    }
    candidates = [
        {"name": "Dr. Amit Sharma", "linkedin_url": "https://linkedin.com/in/amit-sharma-ref", "current_role": "Consultant", "parsed_tags": ["consultant"]},
        {"name": "Lata Vaidyanathan", "linkedin_url": "https://linkedin.com/in/lata-vaidyanathan", "current_role": "Director", "parsed_tags": ["principal", "consultant"]},
    ]
    sem_scores = np.array([0.99, 0.85])
    ranked = rank_candidates(ref_analysis, candidates, sem_scores)

    # Reference candidate must be excluded
    ranked_names = [c["name"] for c in ranked]
    assert "Dr. Amit Sharma" not in ranked_names
    assert "Lata Vaidyanathan" in ranked_names


def test_ranking_sort_order_and_bounds():
    ref_analysis = {
        "name": "Reference Educator",
        "role_family": "School Leadership",
        "school_ecosystem_tags": ["principal"],
        "location": "Delhi NCR",
    }
    candidates = [
        {"name": "Person A", "current_role": "School Principal", "parsed_tags": ["principal"], "location": "Delhi NCR", "multi_school_bool": True},
        {"name": "Person B", "current_role": "Software Developer", "parsed_tags": ["developer"], "location": "London", "multi_school_bool": False},
        {"name": "Person C", "current_role": "Education Consultant", "parsed_tags": ["consultant", "principal"], "location": "Delhi", "multi_school_bool": True},
    ]
    sem_scores = np.array([0.88, 0.12, 0.80])
    ranked = rank_candidates(ref_analysis, candidates, sem_scores)

    assert len(ranked) == 3
    # Check descending sort
    scores = [r["total_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    # Check bounds
    for s in scores:
        assert 0.0 <= s <= 100.0
    # Relevant candidates should beat unrelated developer
    assert ranked[0]["name"] in ["Person A", "Person C"]
    assert ranked[-1]["name"] == "Person B"
