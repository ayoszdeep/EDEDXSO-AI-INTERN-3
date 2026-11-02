"""
profile_analyzer.py
===================
AI and NLP-based profile analysis engine.
Extracts structured indicators, role families, leadership evidence,
and school ecosystem tags from unstructured reference profile inputs.
"""

import json
import re
from typing import Any, Dict, List, Optional, Set


ROLE_FAMILY_PATTERNS = {
    "Education Consulting": [
        r"\bconsultant\b",
        r"\badvisor\b",
        r"\badvisory\b",
        r"\bconsulting\b",
        r"\bstrategist\b",
    ],
    "School Leadership": [
        r"\bprincipal\b",
        r"\bhead of school\b",
        r"\bheadmaster\b",
        r"\bheadmistress\b",
        r"\bvice\s*principal\b",
        r"\brector\b",
        r"\bdean\b",
    ],
    "School Governance": [
        r"\bowner\b",
        r"\bdirector\b",
        r"\btrustee\b",
        r"\bpromoter\b",
        r"\bboard member\b",
        r"\bgovernor\b",
        r"\bchairperson\b",
        r"\bchairman\b",
    ],
    "Education Entrepreneurship": [
        r"\bfounder\b",
        r"\bco-founder\b",
        r"\bentrepreneur\b",
        r"\bedupreneur\b",
        r"\bceo\b",
        r"\bmanaging director\b",
    ],
    "School Network Management": [
        r"\bnetwork director\b",
        r"\bgroup director\b",
        r"\bcluster head\b",
        r"\bregional director\b",
        r"\bacademic director\b",
    ],
}

TAG_PATTERNS = {
    "principal": [
        r"\bprincipal\b",
        r"\bhead of school\b",
        r"\bheadmaster\b",
        r"\bheadmistress\b",
    ],
    "consultant": [
        r"\bconsultant\b",
        r"\badvisor\b",
        r"\badvisory\b",
        r"\bconsulting\b",
    ],
    "school_leadership": [
        r"\bschool leadership\b",
        r"\bprincipal\b",
        r"\bacademic leadership\b",
        r"\bhead of school\b",
        r"\bheadmaster\b",
    ],
    "school_owner": [
        r"\bschool owner\b",
        r"\btrustee\b",
        r"\bpromoter\b",
        r"\bboard of governors\b",
        r"\bchairperson\b",
    ],
    "director": [
        r"\bdirector\b",
        r"\bmanaging director\b",
        r"\bexecutive director\b",
    ],
    "entrepreneur": [
        r"\bfounder\b",
        r"\bco-founder\b",
        r"\bentrepreneur\b",
        r"\bedupreneur\b",
        r"\bstartup\b",
    ],
    "multi_school": [
        r"\bmultiple schools\b",
        r"\bschool network\b",
        r"\bseveral schools\b",
        r"\bgroup of schools\b",
        r"\bacross campuses\b",
        r"\bmulti-school\b",
        r"\bchain of schools\b",
        r"\bcluster\b",
        r"\bpan-india\b",
    ],
    "school_management": [
        r"\bschool management\b",
        r"\bschool operations\b",
        r"\bschool administration\b",
        r"\binstitutional development\b",
    ],
    "curriculum_strategy": [
        r"\bcurriculum\b",
        r"\bpedagogy\b",
        r"\bcbse\b",
        r"\bcisce\b",
        r"\bicse\b",
        r"\bib\b",
        r"\bcambridge\b",
        r"\bacademic strategy\b",
    ],
    "teacher_training": [
        r"\bteacher training\b",
        r"\bprofessional development\b",
        r"\bcapacity building\b",
        r"\bpedagogical mentoring\b",
    ],
    "policy_advocacy": [
        r"\bnep\b",
        r"\bpolicy\b",
        r"\bficci\b",
        r"\bcii\b",
        r"\beducation board\b",
        r"\bministry of education\b",
    ],
}


def extract_experience_years(text: str, explicit_years: Optional[Any] = None) -> Optional[int]:
    """Extract years of experience from explicit value or regex text analysis."""
    if explicit_years is not None and str(explicit_years).strip():
        try:
            return int(float(str(explicit_years).strip()))
        except (ValueError, TypeError):
            pass

    # Pattern: "18 years of experience", "20+ yrs", "15 years in education"
    match = re.search(
        r"(\d{1,2})\+?\s*(?:years?|yrs?)(?:\s*(?:of)?\s*(?:experience|in education|in school|in k-12))?",
        text,
        re.IGNORECASE,
    )
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return None


def extract_role_family(current_role: str, biography: str) -> str:
    """Determine the primary role family based on current role and background."""
    role_lower = current_role.lower()

    # Prioritize current role first
    for family, patterns in ROLE_FAMILY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, role_lower):
                return family

    # Fallback to biography inspection
    bio_lower = biography.lower()
    for family, patterns in ROLE_FAMILY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, bio_lower):
                return family

    return "School Education Professional"


def extract_previous_roles(past_roles_str: str, biography: str) -> List[str]:
    """Identify documented prior roles from past_roles field and biography."""
    detected = set()
    combined = f"{past_roles_str} {biography}".lower()

    role_targets = [
        ("School Principal", [r"\bformer principal\b", r"\bex-principal\b", r"\bprincipal\b", r"\bhead of school\b"]),
        ("Vice Principal", [r"\bvice\s*principal\b"]),
        ("Education Consultant", [r"\bconsultant\b", r"\badvisor\b"]),
        ("School Director", [r"\bschool director\b", r"\bdirector of schools\b"]),
        ("Academic Coordinator", [r"\bacademic coordinator\b", r"\bhead of department\b", r"\bhod\b"]),
        ("School Founder", [r"\bfounder\b", r"\bco-founder\b"]),
        ("Trustee / Board Member", [r"\btrustee\b", r"\bboard member\b"]),
    ]

    for label, patterns in role_targets:
        for pat in patterns:
            if re.search(pat, combined):
                detected.add(label)
                break

    return sorted(list(detected))


def extract_tags(text: str) -> List[str]:
    """Extract normalized school ecosystem tags from combined text."""
    text_lower = text.lower()
    detected_tags = set()

    for tag, patterns in TAG_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                detected_tags.add(tag)
                break

    return sorted(list(detected_tags))


def analyze_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform comprehensive AI/rule-based analysis of a reference profile.

    Parameters:
        profile: Dictionary containing:
            name, current_role, organisation, location, biography,
            past_roles, school_experience, etc.

    Returns:
        Structured analysis dictionary.
    """
    name = str(profile.get("name", "")).strip()
    current_role = str(profile.get("current_role", "")).strip()
    organisation = str(profile.get("organisation", "")).strip()
    location = str(profile.get("location", "")).strip()
    biography = str(profile.get("biography", "")).strip()
    past_roles = str(profile.get("past_roles", "")).strip()
    school_exp = str(profile.get("school_experience", "")).strip()

    combined_text = f"{current_role} {organisation} {biography} {past_roles} {school_exp}"

    # Extract role family and experience
    role_family = extract_role_family(current_role, biography)
    exp_years = extract_experience_years(combined_text, profile.get("experience_years"))
    previous_roles = extract_previous_roles(past_roles, biography)

    # Extract tags
    manual_tags = profile.get("professional_tags", "")
    if isinstance(manual_tags, list):
        parsed_manual = [t.lower().replace(" ", "_") for t in manual_tags]
    else:
        parsed_manual = [t.strip().lower().replace(" ", "_") for t in re.split(r"[;,]", str(manual_tags)) if t.strip()]

    extracted_tags = extract_tags(combined_text)
    all_tags = sorted(list(set(parsed_manual) | set(extracted_tags)))

    # Evidence indicators
    text_lower = combined_text.lower()
    leadership_evidence = any(t in all_tags for t in ["principal", "school_leadership", "director", "school_owner"])
    multi_school_evidence = "multi_school" in all_tags or bool(
        re.search(r"\b(multiple schools|network|several schools|cluster|pan-india|multiple campuses)\b", text_lower)
    )
    consulting_evidence = any(t in all_tags for t in ["consultant", "advisor"]) or bool(
        re.search(r"\b(advises|consulting|guidance to management|strategic advisor)\b", text_lower)
    )
    entrepreneur_evidence = "entrepreneur" in all_tags or bool(re.search(r"\b(founder|co-founder|founded)\b", text_lower))
    decision_maker_evidence = bool(
        leadership_evidence
        or "school_owner" in all_tags
        or re.search(r"\b(trustees|school owners|management teams|board of governors)\b", text_lower)
    )

    # Extract skills / domains
    skills = []
    if "school_management" in all_tags:
        skills.append("School Management & Operations")
    if "school_leadership" in all_tags or leadership_evidence:
        skills.append("Academic & Institutional Leadership")
    if consulting_evidence:
        skills.append("Strategic Education Consulting")
    if multi_school_evidence:
        skills.append("School Network Scaling & Expansion")
    if "curriculum_strategy" in all_tags:
        skills.append("Curriculum & Pedagogical Strategy")
    if "teacher_training" in all_tags:
        skills.append("Teacher Professional Development")
    if entrepreneur_evidence:
        skills.append("Educational Entrepreneurship")

    return {
        "name": name,
        "current_role": current_role,
        "organisation": organisation,
        "location": location,
        "role_family": role_family,
        "industry": "School Education (K-12)",
        "experience_years": exp_years,
        "previous_roles": previous_roles,
        "skills": skills,
        "school_ecosystem_tags": all_tags,
        "leadership_evidence": leadership_evidence,
        "multi_school_evidence": multi_school_evidence,
        "consulting_evidence": consulting_evidence,
        "entrepreneur_evidence": entrepreneur_evidence,
        "decision_maker_evidence": decision_maker_evidence,
        "analysis_summary": (
            f"{name} is an identified {role_family} with approx {exp_years or '10+'} years of K-12 background. "
            f"Evidence shows: Leadership ({'Yes' if leadership_evidence else 'No'}), "
            f"Multi-School Network ({'Yes' if multi_school_evidence else 'No'}), "
            f"Advisory/Consulting ({'Yes' if consulting_evidence else 'No'})."
        ),
    }


def analysis_to_json(analysis: Dict[str, Any]) -> str:
    """Format analysis dictionary as indented JSON."""
    return json.dumps(analysis, indent=2)
