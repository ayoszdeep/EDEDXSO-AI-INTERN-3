"""
test_integration.py
===================
End-to-end integration test validating reference profile analysis,
semantic embedding generation, top 20 candidate discovery, explanations,
and precision measurement.
"""

import pytest
import numpy as np
from data_loader import load_profiles
from profile_analyzer import analyze_profile
from embedding_engine import create_profile_text, calculate_similarity
from ranker import rank_candidates
from explanations import generate_explanation


def test_full_pipeline_with_dr_amit_sharma():
    # 1. Reference Profile
    ref_profile = {
        "name": "Dr. Amit Sharma",
        "current_role": "Former School Principal · Education Consultant",
        "organisation": "Independent School Advisory",
        "location": "Delhi NCR",
        "biography": (
            "18 years of experience in school education. Former principal of two schools. "
            "Advises school owners and management teams. Works with multiple schools on development and operations. "
            "Based in Delhi NCR."
        ),
        "past_roles": "School Principal; Academic Vice Principal",
        "school_experience": "18 years of K-12 institutional governance, principal mentoring, school management advisory.",
        "professional_tags": "principal;consultant;school_leadership;school_management;multi_school",
    }

    # 2. AI Analysis
    analysis = analyze_profile(ref_profile)
    assert analysis["role_family"] == "Education Consulting"
    assert analysis["leadership_evidence"] is True
    assert analysis["multi_school_evidence"] is True
    assert analysis["consulting_evidence"] is True

    # 3. Load Dataset
    df = load_profiles()
    assert len(df) == 50
    candidates = df.to_dict(orient="records")

    # 4. Generate Semantic Embeddings & Similarity
    ref_text = create_profile_text(ref_profile)
    cand_texts = [create_profile_text(c) for c in candidates]
    sims = calculate_similarity(ref_text, cand_texts)

    assert len(sims) == len(candidates)
    assert (sims >= 0.0).all() and (sims <= 1.0).all()

    # 5. Hybrid Ranking
    ranked = rank_candidates(analysis, candidates, sims)
    assert len(ranked) >= 20

    top_20 = ranked[:20]
    assert len(top_20) == 20

    # Ensure scores are strictly sorted descending
    scores = [c["total_score"] for c in top_20]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] >= scores[-1]

    # 6. Explanations and Traceability
    for cand in top_20:
        exp = generate_explanation(analysis, cand)
        assert len(exp["reasons"]) > 0
        assert "score_breakdown" in exp
        assert "source_url" in exp
        assert cand["source_url"].startswith("http")

    # 7. Quality evaluation: Check precision of top 20
    # For Dr. Amit Sharma (former principal + education consultant advising school owners):
    # Leaders who are former/current principals, school consultants, school owners/directors, or multi-school leaders
    relevant_count = 0
    for cand in top_20:
        tags = cand.get("parsed_tags", [])
        is_relevant = (
            "principal" in tags
            or "consultant" in tags
            or "school_leadership" in tags
            or "school_owner" in tags
            or "director" in tags
            or cand.get("multi_school_bool") is True
        )
        if is_relevant:
            relevant_count += 1

    precision_at_20 = relevant_count / 20.0
    print(f"\nPrecision@20 for Dr. Amit Sharma: {precision_at_20 * 100:.1f}% ({relevant_count}/20)")
    assert precision_at_20 >= 0.85, f"Precision@20 was {precision_at_20:.2f}, expected at least 0.85"
