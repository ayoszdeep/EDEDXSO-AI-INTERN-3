"""
test_data.py
============
Unit and validation tests for data_loader and dataset integrity.
"""

import os
import pytest
import pandas as pd
from data_loader import (
    load_profiles,
    get_dataset_summary,
    save_feedback,
    load_feedback,
    REQUIRED_COLUMNS,
)


def test_dataset_file_exists():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "profiles.csv")
    assert os.path.exists(path), f"profiles.csv must exist at {path}"


def test_load_profiles_count():
    df = load_profiles()
    assert len(df) >= 20, f"Expected at least 20 candidates, found {len(df)}"
    # Target 50
    assert len(df) == 50, f"Expected target 50 candidates, found {len(df)}"


def test_required_columns_present():
    df = load_profiles()
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"


def test_no_missing_critical_fields():
    df = load_profiles()
    assert df["name"].str.strip().ne("").all(), "All profiles must have a valid non-empty name"
    assert df["source_url"].str.strip().ne("").all(), "All profiles must have a valid non-empty source_url"
    assert df["current_role"].str.strip().ne("").all(), "All profiles must have a valid non-empty current_role"
    assert df["organisation"].str.strip().ne("").all(), "All profiles must have a valid non-empty organisation"


def test_no_duplicate_candidates():
    df = load_profiles()
    names_clean = df["name"].str.strip().str.lower()
    duplicates = names_clean[names_clean.duplicated()].tolist()
    assert len(duplicates) == 0, f"Found duplicate names: {duplicates}"


def test_dataset_summary_metrics():
    df = load_profiles()
    summary = get_dataset_summary(df)
    assert summary["total_records"] == 50
    assert summary["verified_count"] == 50
    assert len(summary["categories"]) >= 4
    for cat, count in summary["categories"].items():
        assert count > 0, f"Category '{cat}' should have at least 1 record"


def test_human_feedback_recording(tmp_path):
    fb_file = os.path.join(tmp_path, "feedback_test.csv")
    save_feedback("P001", "Lata Vaidyanathan", "Relevant", "Top match for school consulting", feedback_file=fb_file)
    fb_df = load_feedback(feedback_file=fb_file)
    assert len(fb_df) == 1
    assert fb_df.iloc[0]["candidate_id"] == "P001"
    assert fb_df.iloc[0]["feedback_type"] == "Relevant"

    # Append second feedback
    save_feedback("P002", "Dr. Ameeta Mulla Wattal", "Needs Review", "Check location details", feedback_file=fb_file)
    fb_df2 = load_feedback(feedback_file=fb_file)
    assert len(fb_df2) == 2
