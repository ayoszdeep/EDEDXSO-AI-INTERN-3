"""
test_ranker.py
==============
Unit tests for ranking calculation, Jaccard similarity, role compatibility,
network exposure, dynamic weights, and renormalization.
"""

import numpy as np
import pytest
from ranker import (
    tag_similarity,
    calculate_role_similarity,
    calculate_network_score,
    calculate_location_score,
    calculate_final_score,
    compute_candidate_completeness,
    rank_candidates,
    DEFAULT_WEIGHTS,
)


def test_tag_similarity_jaccard():
    assert tag_similarity(["principal", "consultant"], ["principal", "consultant"]) == 1.0
    assert tag_similarity(["principal"], ["software_engineer"]) == 0.0
    sim = tag_similarity(["principal", "consultant"], ["consultant", "director"])
    assert round(sim, 2) == 0.33
    assert tag_similarity([], []) == 0.0


def test_role_similarity():
    assert calculate_role_similarity("School Leadership", "School Leadership") == 1.0
    adj_score = calculate_role_similarity("School Leadership", "Education Consulting")
    assert adj_score >= 0.8
    unrel = calculate_role_similarity("School Leadership", "Software Development")
    assert unrel <= 0.5


def test_network_score():
    cand_yes = {"multi_school_bool": True, "biography": "Advises 10+ schools across Delhi."}
    assert calculate_network_score(cand_yes) == 1.0

    cand_no = {"multi_school_bool": False, "biography": "Principal of single school."}
    assert calculate_network_score(cand_no) == 0.5

    cand_unknown = {"multi_school_bool": None, "biography": "Educator background."}
    assert calculate_network_score(cand_unknown) is None


def test_location_score():
    assert calculate_location_score("Delhi NCR", "Delhi NCR") == 1.0
    assert calculate_location_score("Delhi NCR", "Noida, Delhi NCR") == 1.0
    assert calculate_location_score("Delhi NCR", "Bengaluru, India") == 0.5
    assert calculate_location_score("Delhi NCR", "") is None


def test_renormalization_without_missing():
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
    components = {
        "semantic": 1.0,
        "school": 1.0,
        "role": 1.0,
        "network": None,
        "location": None,
    }
    score, contribs = calculate_final_score(components)
    assert score == 100.0


def test_custom_weights_tuning():
    components = {
        "semantic": 1.0,
        "school": 1.0,
        "role": 1.0,
        "network": 1.0,
        "location": 1.0,
    }
    custom_w = {
        "semantic": 0.60,
        "school": 0.10,
        "role": 0.10,
        "network": 0.10,
        "location": 0.10,
    }
    score, contribs = calculate_final_score(components, weights_dict=custom_w)
    assert score == 100.0
    assert contribs["semantic"] == 60.0
    assert contribs["school"] == 10.0


def test_candidate_completeness_calculation():
    cand_full = {
        "name": "Dr. Full Record",
        "current_role": "Director",
        "organisation": "Sample Org",
        "location": "Delhi",
        "biography": "Extensive bio here.",
        "experience_years": 20,
        "multi_school_bool": True,
        "decision_maker_bool": True,
        "source_url": "https://example.com",
        "source_date": "2026-01-01",
    }
    assert compute_candidate_completeness(cand_full) == 100.0

    cand_sparse = {
        "name": "Dr. Sparse",
        "current_role": "Principal",
        "organisation": "School",
        "source_url": "https://example.com",
    }
    assert compute_candidate_completeness(cand_sparse) < 60.0


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
    scores = [r["total_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    for s in scores:
        assert 0.0 <= s <= 100.0
    assert ranked[0]["name"] in ["Person A", "Person C"]
    assert ranked[-1]["name"] == "Person B"
