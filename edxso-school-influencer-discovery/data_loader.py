"""
data_loader.py
==============
Data loading, validation, sanitization, and summary module for the
EDXSO School Influencer Discovery Tool.
Includes data health diagnostics, structured logging, and persistent feedback tracking.
"""

import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from logger_config import setup_logger

logger = setup_logger("data_loader")

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "data", "profiles.csv"
)
DEFAULT_FEEDBACK_PATH = os.path.join(
    os.path.dirname(__file__), "data", "feedback.csv"
)

REQUIRED_COLUMNS = [
    "id",
    "name",
    "current_role",
    "organisation",
    "location",
    "biography",
    "professional_tags",
    "source_url",
]


def normalize_boolean(value: Any) -> Optional[bool]:
    """Convert flexible string/bool input to Optional[bool]."""
    if pd.isna(value) or value is None:
        return None
    val_str = str(value).strip().lower()
    if val_str in ("yes", "true", "1", "t", "y"):
        return True
    if val_str in ("no", "false", "0", "f", "n"):
        return False
    return None


def parse_tags(tag_str: Any) -> List[str]:
    """Parse delimited tags string into a cleaned list of lower-case tags."""
    if pd.isna(tag_str) or not tag_str:
        return []
    raw_tags = re.split(r"[;,]", str(tag_str))
    cleaned = [t.strip().lower().replace(" ", "_") for t in raw_tags if t.strip()]
    return sorted(list(set(cleaned)))


def load_profiles(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load, validate, and clean professional profiles from CSV.
    """
    start_time = time.time()
    path = csv_path or DEFAULT_DATA_PATH
    if not os.path.exists(path):
        logger.error("Candidate dataset not found at: %s", path)
        raise FileNotFoundError(f"Candidate dataset not found at: {path}")

    df = pd.read_csv(path, encoding="utf-8")

    # Validate schema
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.critical("Dataset missing required columns: %s", missing_cols)
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    initial_count = len(df)
    df = df[df["name"].notna() & (df["name"].str.strip() != "")].copy()
    df = df[df["source_url"].notna() & (df["source_url"].str.strip() != "")].copy()

    if "data_status" in df.columns:
        df = df[df["data_status"].fillna("").str.lower() != "excluded"].copy()

    # Deduplicate by Name (case-insensitive) and non-empty LinkedIn URL
    df["name_clean"] = df["name"].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset=["name_clean"], keep="first").copy()
    df = df.drop(columns=["name_clean"])

    if "linkedin_url" in df.columns:
        has_linkedin = df["linkedin_url"].notna() & (df["linkedin_url"].str.strip() != "")
        li_dupes = df[has_linkedin].duplicated(subset=["linkedin_url"], keep="first")
        if li_dupes.any():
            drop_indices = df[has_linkedin][li_dupes].index
            df = df.drop(drop_indices)

    # Normalize fields
    df["experience_years"] = pd.to_numeric(df.get("experience_years"), errors="coerce")
    df["multi_school_bool"] = df.get("multi_school_experience", "unknown").apply(normalize_boolean)
    df["decision_maker_bool"] = df.get("decision_maker", "unknown").apply(normalize_boolean)
    df["parsed_tags"] = df["professional_tags"].apply(parse_tags)

    text_fields = ["current_role", "organisation", "location", "biography", "past_roles", "school_experience", "linkedin_url"]
    for col in text_fields:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    logger.info("Loaded and validated %d profiles (from %d rows) in %.2f ms", len(df), initial_count, elapsed_ms)

    return df.reset_index(drop=True)


def validate_dataset_health(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform dataset health diagnostics and completeness audit."""
    total = len(df)
    missing_experience = df["experience_years"].isna().sum()
    missing_location = df["location"].eq("").sum()
    verified_records = df["data_status"].str.lower().eq("verified").sum() if "data_status" in df.columns else total
    source_valid_count = df["source_url"].str.startswith("http").sum()

    return {
        "total_records": total,
        "verified_rate": round((verified_records / total) * 100, 1) if total else 0,
        "source_validity_rate": round((source_valid_count / total) * 100, 1) if total else 0,
        "missing_experience_count": int(missing_experience),
        "missing_location_count": int(missing_location),
        "health_score": round(((verified_records + source_valid_count) / (2 * total)) * 100, 1) if total else 0,
    }


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute key metrics and categorical breakdowns of the candidate dataset."""
    total = len(df)
    all_tags = [t for tags in df["parsed_tags"] for t in tags]
    tag_counts = pd.Series(all_tags).value_counts().to_dict()

    categories = {
        "Former / Current Principals": df["parsed_tags"].apply(lambda tags: "principal" in tags or "head_of_school" in tags).sum(),
        "Education Consultants & Advisors": df["parsed_tags"].apply(lambda tags: "consultant" in tags or "advisor" in tags).sum(),
        "School Owners, Directors & Trustees": df["parsed_tags"].apply(lambda tags: "school_owner" in tags or "director" in tags or "trustee" in tags).sum(),
        "Education Entrepreneurs & Founders": df["parsed_tags"].apply(lambda tags: "entrepreneur" in tags or "founder" in tags).sum(),
        "Multi-School & Network Leaders": df["multi_school_bool"].eq(True).sum(),
    }

    verified_count = (df["data_status"].fillna("").str.lower() == "verified").sum() if "data_status" in df.columns else total
    locations = df["location"].value_counts().head(5).to_dict()
    health = validate_dataset_health(df)

    return {
        "total_records": total,
        "categories": categories,
        "verified_count": int(verified_count),
        "top_locations": locations,
        "unique_tags_count": len(tag_counts),
        "avg_experience_years": float(df["experience_years"].dropna().mean()) if "experience_years" in df.columns and not df["experience_years"].dropna().empty else 0.0,
        "health": health,
    }


def save_feedback(
    candidate_id: str,
    candidate_name: str,
    feedback_type: str,
    comment: Optional[str] = "",
    feedback_file: Optional[str] = None
) -> None:
    """Record human feedback on discovered candidates to support continuous learning."""
    path = feedback_file or DEFAULT_FEEDBACK_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    new_row = {
        "timestamp": datetime.now().isoformat(),
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "feedback_type": feedback_type,
        "comment": comment or "",
    }

    if not os.path.exists(path):
        pd.DataFrame([new_row]).to_csv(path, index=False)
    else:
        pd.DataFrame([new_row]).to_csv(path, mode="a", header=False, index=False)

    logger.info("Logged evaluator feedback for %s (%s): %s", candidate_name, candidate_id, feedback_type)


def load_feedback(feedback_file: Optional[str] = None) -> pd.DataFrame:
    """Load previously recorded human feedback."""
    path = feedback_file or DEFAULT_FEEDBACK_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp", "candidate_id", "candidate_name", "feedback_type", "comment"])
    return pd.read_csv(path)
