import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader import load_profiles, get_dataset_summary, save_feedback, load_feedback
from profile_analyzer import analyze_profile, analysis_to_json
from embedding_engine import create_profile_text, calculate_similarity
from ranker import rank_candidates, calculate_final_score
from explanations import generate_explanation

def run_smoke():
    # 1. Dataset verification
    df = load_profiles()
    summary = get_dataset_summary(df)
    print(f"Dataset check: {len(df)} profiles loaded.")
    print(f"Category counts: {summary['categories']}")

    # 2. Reference test
    ref = {
        'name': 'Dr. Amit Sharma',
        'current_role': 'Former School Principal · Education Consultant',
        'organisation': 'EduConsult Advisory',
        'location': 'Delhi NCR',
        'biography': '18 years in school education. Former principal of two schools. Advises school owners and management teams. Works with multiple schools on development and operations. Based in Delhi NCR.',
        'past_roles': 'School Principal; Vice Principal',
        'school_experience': 'K-12 institutional governance, principal mentoring, school management advisory across northern India.',
        'professional_tags': 'principal;consultant;school_leadership;school_management;multi_school'
    }

    analysis = analyze_profile(ref)
    print(f"Analysis check: role_family={analysis['role_family']}, leadership={analysis['leadership_evidence']}, multi_school={analysis['multi_school_evidence']}")

    # 3. Discovery check
    candidates = df.to_dict(orient='records')
    ref_text = create_profile_text(ref)
    cand_texts = [create_profile_text(c) for c in candidates]
    sims = calculate_similarity(ref_text, cand_texts)
    ranked = rank_candidates(analysis, candidates, sims)
    top_20 = ranked[:20]

    print(f"Ranking check: Discovered {len(ranked)} eligible candidates. Top 20 extracted.")
    print("\nTop 5 Discovered Influencers:")
    for idx, cand in enumerate(top_20[:5], 1):
        exp = generate_explanation(analysis, cand)
        print(f"{idx}. {cand['name']} ({cand['total_score']}/100) - {cand['current_role']} at {cand['organisation']}")
        print(f"   Breakdown: {exp['score_breakdown']}")
        print(f"   Source: {cand['source_url']}")

    print("\nSMOKE TEST COMPLETED SUCCESSFULLY!")

if __name__ == '__main__':
    run_smoke()
