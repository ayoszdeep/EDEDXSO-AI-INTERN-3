"""
app.py
======
Main Streamlit Application for EDXSO School Influencer Discovery Tool.
AI-powered school education professional discovery, hybrid semantic ranking,
and explainable provenance dashboard.
"""

import json
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import (
    load_profiles,
    get_dataset_summary,
    save_feedback,
    load_feedback,
)
from profile_analyzer import analyze_profile, analysis_to_json
from embedding_engine import create_profile_text, calculate_similarity
from ranker import rank_candidates, WEIGHTS
from explanations import generate_explanation

# Page Configuration
st.set_page_config(
    page_title="EDXSO | School Influencer Discovery Tool",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .candidate-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .score-badge-high {
        background-color: #DCFCE7;
        color: #15803D;
        font-weight: 700;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.9rem;
    }
    .score-badge-med {
        background-color: #FEF9C3;
        color: #A16207;
        font-weight: 700;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.9rem;
    }
    .score-badge-low {
        background-color: #FEE2E2;
        color: #B91C1C;
        font-weight: 700;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.9rem;
    }
    .tag-pill {
        display: inline-block;
        background: #EFF6FF;
        color: #1D4ED8;
        padding: 0.2rem 0.5rem;
        margin: 0.15rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper function to cache loaded data
@st.cache_data
def get_cached_dataset():
    return load_profiles()


# Default Reference Profile (Illustrative Example from Assignment)
DEFAULT_REFERENCE = {
    "name": "Dr. Amit Sharma",
    "linkedin_url": "https://www.linkedin.com/in/dr-amit-sharma-illustrative",
    "current_role": "Former School Principal · Education Consultant",
    "organisation": "EduConsult Management Advisory",
    "location": "Delhi NCR, India",
    "biography": (
        "18 years of experience in school education. Former principal of two schools. "
        "Advises school owners and management teams. Works with multiple schools on development and operations. "
        "Based in Delhi NCR."
    ),
    "past_roles": "School Principal (two terms); Academic Vice Principal",
    "school_experience": (
        "18 years across K-12 administration, CBSE affiliation, leadership development, "
        "and school management advisory across northern India."
    ),
    "professional_tags": "principal;consultant;school_leadership;school_management;multi_school;advisor",
}

# Initialize Session State
if "ref_profile" not in st.session_state:
    st.session_state["ref_profile"] = DEFAULT_REFERENCE.copy()

if "analysis" not in st.session_state:
    st.session_state["analysis"] = analyze_profile(st.session_state["ref_profile"])

if "discovery_results" not in st.session_state:
    st.session_state["discovery_results"] = None

# Load Candidate Profiles
try:
    df_candidates = get_cached_dataset()
    candidates_list = df_candidates.to_dict(orient="records")
    summary = get_dataset_summary(df_candidates)
except Exception as e:
    st.error(f"Error loading verified dataset: {e}")
    st.stop()

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-header">🎓 EDXSO School Influencer Discovery Tool</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-powered discovery & hybrid semantic ranking for school-education professionals, consultants, and decision-makers.</div>',
        unsafe_allow_html=True,
    )
with col_h2:
    st.caption("AI Intern Assignment MVP")
    st.caption("Model: `all-MiniLM-L6-v2` | DPDP 2025 Ready")

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/000000/school.png", width=60)
st.sidebar.title("Navigation & Controls")

nav_choice = st.sidebar.radio(
    "Go to View:",
    [
        "📊 Dashboard Overview",
        "👤 Reference Profile Input & AI Analysis",
        "🔍 Discovered Influencers (Top 20+)",
        "🗂️ Explore Candidate Dataset",
        "⚖️ Provenance, Privacy & Technical Note",
    ],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Quick Reference Selector")
if st.sidebar.button("Pre-fill Dr. Amit Sharma (Demo Example)", use_container_width=True):
    st.session_state["ref_profile"] = DEFAULT_REFERENCE.copy()
    st.session_state["analysis"] = analyze_profile(st.session_state["ref_profile"])
    st.session_state["discovery_results"] = None
    st.sidebar.success("Loaded Dr. Amit Sharma illustrative profile!")

st.sidebar.divider()
st.sidebar.info(
    "**LinkedIn Compliance:**\n"
    "This tool relies strictly on a verified, source-linked public dataset. "
    "No scraping, bot extensions, or authentication bypass are performed."
)


# ==============================================================================
# VIEW 1: DASHBOARD OVERVIEW
# ==============================================================================
if nav_choice == "📊 Dashboard Overview":
    st.subheader("System Overview & Verified Metrics")

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{summary['total_records']}</div>
                <div class="metric-label">Verified Profiles</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(summary['categories'])}</div>
                <div class="metric-label">Professional Categories</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">20+</div>
                <div class="metric-label">Target Discovery Results</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">45/20/15/15/5</div>
                <div class="metric-label">Hybrid Ranking Weights</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    col_cat, col_work = st.columns([1, 1])
    with col_cat:
        st.markdown("#### 📂 Verified Dataset Breakdown")
        cat_df = pd.DataFrame(
            list(summary["categories"].items()),
            columns=["Category", "Verified Candidates"],
        )
        st.dataframe(cat_df, hide_index=True, use_container_width=True)

        st.markdown("#### 📍 Top Geographic Hubs")
        loc_df = pd.DataFrame(
            list(summary["top_locations"].items()),
            columns=["Region / City", "Candidates"],
        )
        st.dataframe(loc_df, hide_index=True, use_container_width=True)

    with col_work:
        st.markdown("#### 🔄 Discovery Workflow Pipeline")
        st.markdown(
            """
            1. **Reference Input**: Supply a reference professional's background, past leadership roles, and school experience.
            2. **AI Profile Analysis**: Extract canonical role family, previous professions, school ecosystem tags, and decision-maker indicators.
            3. **Candidate Vector Retrieval**: Generate dense semantic embeddings via Sentence Transformers (`all-MiniLM-L6-v2`) and calculate cosine similarity across candidate biographies.
            4. **Hybrid Relevance Scoring**: Apply weighted multi-factor ranking combining:
               - **Semantic Similarity (45%)**
               - **School Ecosystem Tags (20%)**
               - **Role & Experience Match (15%)**
               - **Multi-School Exposure (15%)**
               - **Location Alignment (5%)**
            5. **Explainability & Provenance**: Render ranked candidate cards featuring evidence-based justifications and original public source URLs.
            """
        )

        st.markdown("#### 🚀 Ready to Run Discovery?")
        if st.button("Proceed to Reference Input & Analysis ➡️", type="primary"):
            st.session_state["nav_choice"] = "👤 Reference Profile Input & AI Analysis"
            st.rerun()


# ==============================================================================
# VIEW 2: REFERENCE PROFILE INPUT & AI ANALYSIS
# ==============================================================================
elif nav_choice == "👤 Reference Profile Input & AI Analysis":
    st.subheader("Reference Profile Input & AI Analysis")
    st.markdown(
        "Enter the professional background of the reference school leader. "
        "The AI will extract structured indicators and role taxonomies for discovery."
    )

    with st.form("reference_form"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            ref_name = st.text_input("Full Name *", value=st.session_state["ref_profile"].get("name", ""))
            ref_role = st.text_input("Current Position / Job Title *", value=st.session_state["ref_profile"].get("current_role", ""))
            ref_org = st.text_input("Current Organisation *", value=st.session_state["ref_profile"].get("organisation", ""))
            ref_loc = st.text_input("Location (City / Region) *", value=st.session_state["ref_profile"].get("location", ""))
            ref_li = st.text_input("LinkedIn Profile URL (Optional reference link)", value=st.session_state["ref_profile"].get("linkedin_url", ""))

        with col_f2:
            ref_bio = st.text_area(
                "Professional Biography *",
                value=st.session_state["ref_profile"].get("biography", ""),
                height=110,
                help="Describe their career path, leadership responsibilities, and school education work.",
            )
            ref_past = st.text_input(
                "Previous Positions / Past Roles",
                value=st.session_state["ref_profile"].get("past_roles", ""),
                help="e.g. Former Principal of two schools, Vice Principal, Academic Director",
            )
            ref_exp = st.text_area(
                "School-Related Experience Details",
                value=st.session_state["ref_profile"].get("school_experience", ""),
                height=90,
                help="Details on school networks, governance, management, or advisory.",
            )

        submitted = st.form_submit_button("🔍 Run AI Profile Analysis", type="primary")
        if submitted:
            updated_ref = {
                "name": ref_name,
                "current_role": ref_role,
                "organisation": ref_org,
                "location": ref_loc,
                "linkedin_url": ref_li,
                "biography": ref_bio,
                "past_roles": ref_past,
                "school_experience": ref_exp,
            }
            st.session_state["ref_profile"] = updated_ref
            st.session_state["analysis"] = analyze_profile(updated_ref)
            st.session_state["discovery_results"] = None
            st.success("AI Profile Analysis updated successfully!")

    # Display AI Analysis Results
    st.divider()
    st.markdown("### 🤖 Extracted AI Representation")

    analysis = st.session_state["analysis"]

    col_a1, col_a2 = st.columns([1, 1])
    with col_a1:
        st.markdown("#### Structured Indicators")
        st.write(f"**Primary Role Family:** `{analysis.get('role_family')}`")
        st.write(f"**Detected Prior Roles:** `{', '.join(analysis.get('previous_roles', [])) or 'None explicitly listed'}`")
        st.write(f"**Estimated Experience:** `{analysis.get('experience_years') or '15+'} Years`")
        st.write(f"**Industry Domain:** `{analysis.get('industry')}`")

        st.markdown("#### Verified Evidence Flags")
        c_e1, c_e2, c_e3 = st.columns(3)
        with c_e1:
            st.metric("Leadership", "Verified" if analysis.get("leadership_evidence") else "Not detected")
        with c_e2:
            st.metric("Multi-School Network", "Verified" if analysis.get("multi_school_evidence") else "Not detected")
        with c_e3:
            st.metric("Consulting / Advisory", "Verified" if analysis.get("consulting_evidence") else "Not detected")

        st.markdown("#### Extracted Ecosystem Tags")
        tags_html = "".join([f'<span class="tag-pill">{t}</span>' for t in analysis.get("school_ecosystem_tags", [])])
        st.markdown(tags_html, unsafe_allow_html=True)

    with col_a2:
        st.markdown("#### Raw AI Analysis (JSON)")
        st.code(analysis_to_json(analysis), language="json")

    st.write("")
    if st.button("Proceed to Influencer Discovery & Ranking ➡️", type="primary"):
        st.session_state["nav_choice"] = "🔍 Discovered Influencers (Top 20+)"
        st.rerun()


# ==============================================================================
# VIEW 3: DISCOVERY & RANKING (TOP 20+)
# ==============================================================================
elif nav_choice == "🔍 Discovered Influencers (Top 20+)":
    st.subheader("Candidate Discovery & Hybrid Ranking")
    analysis = st.session_state["analysis"]

    st.markdown(
        f"Discovering professionals similar to **{analysis.get('name', 'Reference Profile')}** "
        f"({analysis.get('role_family')} · {analysis.get('location')})."
    )

    # Compute or load discovery results
    if st.session_state["discovery_results"] is None:
        with st.spinner("Calculating semantic embeddings and hybrid ranking scores..."):
            ref_text = create_profile_text(st.session_state["ref_profile"])
            cand_texts = [create_profile_text(c) for c in candidates_list]
            similarities = calculate_similarity(ref_text, cand_texts)
            ranked_results = rank_candidates(analysis, candidates_list, similarities)
            st.session_state["discovery_results"] = ranked_results

    ranked_candidates = st.session_state["discovery_results"]

    # Filter Controls
    with st.expander("⚙️ Refine Results & Interactive Filters", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            all_families = sorted(list(set(c["role_family"] for c in ranked_candidates)))
            selected_families = st.multiselect("Role Family Filter", all_families, default=all_families)
        with f_col2:
            all_locs = sorted(list(set(c["location"] for c in ranked_candidates)))
            selected_locs = st.multiselect("Location Filter", all_locs, default=[])
        with f_col3:
            min_score_filter = st.slider("Minimum Relevance Score", min_value=0, max_value=95, value=50, step=5)
        with f_col4:
            top_k = st.selectbox("Number of Results to Display", [20, 25, 30, 40, 50], index=0)

    # Apply filters
    filtered_results = [
        c for c in ranked_candidates
        if c["role_family"] in selected_families
        and (not selected_locs or c["location"] in selected_locs)
        and c["total_score"] >= min_score_filter
    ]
    display_candidates = filtered_results[:top_k]

    st.markdown(
        f"**Showing {len(display_candidates)} of {len(ranked_candidates)} eligible candidates** "
        f"(Filtered from 50 verified dataset records; Reference person excluded)."
    )

    # Export Button
    if display_candidates:
        export_df = pd.DataFrame([
            {
                "Rank": idx + 1,
                "ID": c["id"],
                "Name": c["name"],
                "Total Relevance Score": c["total_score"],
                "Current Role": c["current_role"],
                "Organisation": c["organisation"],
                "Location": c["location"],
                "Role Family": c["role_family"],
                "Semantic Score (pts)": c.get("contributions", {}).get("semantic", 0),
                "School Tags Score (pts)": c.get("contributions", {}).get("school", 0),
                "Role Match (pts)": c.get("contributions", {}).get("role", 0),
                "Multi-School Reach (pts)": c.get("contributions", {}).get("network", 0),
                "Location Score (pts)": c.get("contributions", {}).get("location", 0),
                "LinkedIn URL": c.get("linkedin_url", ""),
                "Source Evidence URL": c.get("source_url", ""),
                "Source Date": c.get("source_date", ""),
            }
            for idx, c in enumerate(display_candidates)
        ])
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Top Discovered Candidates (CSV)",
            data=csv_bytes,
            file_name=f"edxso_discovered_influencers_{analysis.get('name', 'reference').replace(' ', '_')}.csv",
            mime="text/csv",
        )

    # Render Candidate Cards
    for rank_idx, cand in enumerate(display_candidates, start=1):
        total_score = cand["total_score"]
        if total_score >= 80:
            badge_class = "score-badge-high"
        elif total_score >= 65:
            badge_class = "score-badge-med"
        else:
            badge_class = "score-badge-low"

        explanation = generate_explanation(analysis, cand)

        with st.container():
            st.markdown(
                f"""
                <div class="candidate-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span style="font-size: 1.25rem; font-weight: 700; color: #1E293B;">
                                #{rank_idx}. {cand['name']}
                            </span>
                            <div style="color: #475569; font-weight: 500; font-size: 0.95rem; margin-top: 0.2rem;">
                                {cand['current_role']} · <em>{cand['organisation']}</em>
                            </div>
                            <div style="color: #64748B; font-size: 0.85rem; margin-top: 0.2rem;">
                                📍 {cand['location']} &nbsp;|&nbsp; 🏷️ Role Family: <strong>{cand['role_family']}</strong>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <span class="{badge_class}">{total_score} / 100</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(f"🔍 View Match Explanation & Source Evidence for {cand['name']}"):
                col_exp1, col_exp2 = st.columns([3, 2])
                with col_exp1:
                    st.markdown(f"**Verdict:** {explanation['verdict']}")
                    st.markdown("**Why this person matches:**")
                    for reason in explanation["reasons"]:
                        st.markdown(f"- {reason}")

                    st.markdown("**Professional Background:**")
                    st.write(cand.get("biography", ""))
                    if cand.get("past_roles"):
                        st.caption(f"**Past Positions:** {cand.get('past_roles')}")
                    if cand.get("school_experience"):
                        st.caption(f"**School Ecosystem Experience:** {cand.get('school_experience')}")

                with col_exp2:
                    st.markdown("**📊 Score Contribution Breakdown:**")
                    contribs = cand.get("contributions", {})
                    breakdown_data = {
                        "Component": ["AI Semantic (45%)", "School Tags (20%)", "Role Match (15%)", "Multi-School (15%)", "Location (5%)"],
                        "Points Awarded": [
                            f"{contribs.get('semantic', 0)} pts",
                            f"{contribs.get('school', 0)} pts",
                            f"{contribs.get('role', 0)} pts",
                            f"{contribs.get('network', 0)} pts",
                            f"{contribs.get('location', 0)} pts",
                        ],
                    }
                    st.table(pd.DataFrame(breakdown_data))

                    st.markdown("**🔗 Verified Evidence & Links:**")
                    st.markdown(f"• **Source Proof:** [View Official Verification URL]({cand['source_url']})")
                    if cand.get("linkedin_url"):
                        st.markdown(f"• **LinkedIn:** [Public Profile Record]({cand['linkedin_url']})")
                    st.caption(f"Verified Date: {cand.get('source_date', 'Recent')} | Status: {cand.get('data_status', 'Verified')}")

                    # Human Feedback Section
                    st.divider()
                    st.markdown("💬 **Evaluator Feedback Loop**")
                    fb_choice = st.radio(
                        "Relevance Verdict:",
                        ["Relevant", "Needs Review", "Not Relevant"],
                        key=f"fb_radio_{cand['id']}",
                        horizontal=True,
                    )
                    fb_comment = st.text_input("Optional note:", key=f"fb_note_{cand['id']}", placeholder="e.g. Excellent principal transition match")
                    if st.button("Save Feedback", key=f"fb_btn_{cand['id']}"):
                        save_feedback(cand["id"], cand["name"], fb_choice, fb_comment)
                        st.success(f"Feedback saved for {cand['name']}!")


# ==============================================================================
# VIEW 4: DATASET BROWSER
# ==============================================================================
elif nav_choice == "🗂️ Explore Candidate Dataset":
    st.subheader("Verified Professional Dataset Explorer")
    st.markdown(
        "Browse the curated, source-verified dataset of 50 school-education leaders, "
        "consultants, directors, and entrepreneurs."
    )

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        search_query = st.text_input("Search dataset by name, organisation, role, or keyword:", "")
    with col_s2:
        tag_filter = st.selectbox(
            "Filter by tag:",
            ["All Tags"] + sorted(list(set(t for tags in df_candidates["parsed_tags"] for t in tags))),
        )

    view_df = df_candidates.copy()
    if search_query:
        q = search_query.lower()
        view_df = view_df[
            view_df["name"].str.lower().str.contains(q)
            | view_df["current_role"].str.lower().str.contains(q)
            | view_df["organisation"].str.lower().str.contains(q)
            | view_df["biography"].str.lower().str.contains(q)
        ]

    if tag_filter != "All Tags":
        view_df = view_df[view_df["parsed_tags"].apply(lambda tags: tag_filter in tags)]

    st.markdown(f"**Found {len(view_df)} records:**")
    st.dataframe(
        view_df[[
            "id", "name", "current_role", "organisation", "location",
            "experience_years", "multi_school_experience", "decision_maker", "source_url"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("#### 📝 Collected Human Feedback History")
    feedback_df = load_feedback()
    if not feedback_df.empty:
        st.dataframe(feedback_df, use_container_width=True, hide_index=True)
    else:
        st.info("No evaluator feedback recorded yet. Use the feedback buttons on discovered candidate cards!")


# ==============================================================================
# VIEW 5: PROVENANCE, PRIVACY & TECHNICAL NOTE
# ==============================================================================
elif nav_choice == "⚖️ Provenance, Privacy & Technical Note":
    st.subheader("Data Privacy, Legal Compliance & Technical Note")

    st.markdown(
        """
        ### 1. LinkedIn Data Compliance & Access Boundaries
        - **Zero Scraping Policy:** This system strictly adheres to LinkedIn's User Agreement and terms of service.
          It **does not** utilize Selenium, Playwright, Puppeteer, BeautifulSoup, or unauthorized browser extensions
          to scrape protected LinkedIn profiles.
        - **Manually Permitted Reference:** Users provide the reference professional's background directly or through
          approved public records.
        - **Authorised Public Dataset:** Candidate profiles are collected exclusively from lawfully accessible public
          sources, such as official school websites, conference speaker directories (FICCI ARISE, ScooNews,
          EducationWorld, CII), and published institutional biographies.
        
        ### 2. Digital Personal Data Protection Act (DPDP Act 2023 / Rules 2025)
        - **Purpose Limitation:** Information is used strictly for professional influencer discovery and school-network mapping.
        - **Data Minimisation:** No private personal numbers, personal home addresses, or private communications are collected.
        - **Traceability & Accuracy:** Every profile retains an explicit `source_url` and `source_date` for auditability.
        - **Right to Erasure / Correction:** Records marked as `excluded` in `data_status` are permanently purged from discovery.

        ### 3. AI / ML Architecture & Hybrid Ranking Formula
        $$\\text{Total Score} = \\frac{0.45 S_{\\text{semantic}} + 0.20 S_{\\text{school}} + 0.15 S_{\\text{role}} + 0.15 S_{\\text{network}} + 0.05 S_{\\text{location}}}{\\sum_{k \\in \\text{available}} w_k} \\times 100$$
        
        - **Semantic Similarity ($45\\%$):** Dense embeddings generated by `sentence-transformers/all-MiniLM-L6-v2`.
          Captures conceptual nuance (e.g. former principal transitioning to school governance advisor).
        - **School Ecosystem Tags ($20\\%$):** Jaccard index over normalized professional taxonomy tags.
        - **Role & Experience Proximity ($15\\%$):** Cross-role transition compatibility matrix with experience delta adjustments.
        - **Multi-School Exposure ($15\\%$):** Quantified institutional reach (network leaders vs single campus).
        - **Location Relevance ($5\\%$):** Regional and metro education hub alignment.
        - **Graceful Renormalization:** If a field is unknown, it is marked as `None` and remaining component weights
          are renormalized without false negative penalization.
        """
    )
