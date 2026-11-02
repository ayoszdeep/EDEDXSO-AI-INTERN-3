"""
test_analyzer.py
================
Unit tests for AI profile analysis and entity extraction logic.
"""

import pytest
from profile_analyzer import (
    analyze_profile,
    extract_experience_years,
    extract_role_family,
    extract_tags,
    analysis_to_json,
)


def test_principal_role_detection():
    profile = {
        "name": "Test Principal",
        "current_role": "School Principal",
        "organisation": "Delhi Public School",
        "biography": "Served as principal for 12 years leading CBSE curriculum and teacher development.",
    }
    analysis = analyze_profile(profile)
    assert analysis["role_family"] == "School Leadership"
    assert "principal" in analysis["school_ecosystem_tags"]
    assert analysis["leadership_evidence"] is True


def test_consultant_role_detection():
    profile = {
        "name": "Test Consultant",
        "current_role": "Education Consultant",
        "organisation": "Strategic School Advisory",
        "biography": "Advises school management boards and trustees on academic vision and expansion.",
    }
    analysis = analyze_profile(profile)
    assert analysis["role_family"] == "Education Consulting"
    assert "consultant" in analysis["school_ecosystem_tags"]
    assert analysis["consulting_evidence"] is True


def test_school_owner_and_trustee_detection():
    profile = {
        "name": "Test Trustee",
        "current_role": "Managing Trustee & Director",
        "organisation": "Heritage Educational Trust",
        "biography": "Promoter and board member of five private schools across northern India.",
    }
    analysis = analyze_profile(profile)
    assert analysis["role_family"] == "School Governance"
    assert analysis["decision_maker_evidence"] is True


def test_education_entrepreneur_detection():
    profile = {
        "name": "Test Founder",
        "current_role": "Founder & CEO",
        "organisation": "NextGen School EdTech",
        "biography": "Founded classroom learning platform reaching 500 schools nationwide.",
    }
    analysis = analyze_profile(profile)
    assert analysis["role_family"] == "Education Entrepreneurship"
    assert "entrepreneur" in analysis["school_ecosystem_tags"]
    assert analysis["entrepreneur_evidence"] is True


def test_multi_school_evidence_detection():
    profile = {
        "name": "Multi School Leader",
        "current_role": "Regional Director",
        "organisation": "Podar Education Network",
        "biography": "Directs academic operations across multiple schools and 15 campuses.",
    }
    analysis = analyze_profile(profile)
    assert analysis["multi_school_evidence"] is True
    assert "multi_school" in analysis["school_ecosystem_tags"]


def test_experience_years_extraction():
    assert extract_experience_years("18 years of experience in school education.") == 18
    assert extract_experience_years("Over 25+ years in education leadership.") == 25
    assert extract_experience_years("Text without years", explicit_years=30) == 30
    assert extract_experience_years("No numbers anywhere.") is None


def test_dr_amit_sharma_reference_profile():
    ref = {
        "name": "Dr. Amit Sharma",
        "current_role": "Former School Principal · Education Consultant",
        "organisation": "Independent Education Advisory",
        "location": "Delhi NCR",
        "biography": "18 years of experience in school education. Former principal of two schools. Advises school owners and management teams. Works with multiple schools on development and operations. Based in Delhi NCR.",
        "past_roles": "School Principal; Vice Principal",
        "school_experience": "18 years across K-12 administration and management consulting.",
    }
    analysis = analyze_profile(ref)
    assert analysis["role_family"] == "Education Consulting"
    assert "School Principal" in analysis["previous_roles"]
    assert analysis["experience_years"] == 18
    assert analysis["leadership_evidence"] is True
    assert analysis["consulting_evidence"] is True
    assert analysis["multi_school_evidence"] is True
    assert analysis["decision_maker_evidence"] is True


def test_minimal_and_empty_profile_handling():
    analysis = analyze_profile({})
    assert analysis["name"] == ""
    assert isinstance(analysis["school_ecosystem_tags"], list)
    json_out = analysis_to_json(analysis)
    assert "industry" in json_out
