from rag_pipeline import screen_resume


# ============================================================
# RESUME PDF
# ============================================================

PDF_PATH = (
    "sample_resumes/"
    "Sumit_Naresh_Ghodke_Sep_2026_Resume.pdf"
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

JOB_DESCRIPTION = """
We are looking for a Data Scientist.

Required skills:

- Python
- SQL
- Machine Learning
- Statistics
- Scikit-learn
- Predictive Modeling
- Data Analysis

The candidate should have experience building
machine learning models and analyzing business data.
"""


# ============================================================
# RUN SCREENING
# ============================================================

result = screen_resume(
    PDF_PATH,
    JOB_DESCRIPTION
)


evaluation = result["evaluation"]

score_data = result["score_data"]


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n==============================")
print("RESUME SCREENING RESULT")
print("==============================")


print(
    f"\nMatch Score: "
    f"{score_data['score']}/100"
)


print(
    f"Matching: "
    f"{score_data['matching']}/"
    f"{score_data['total']}"
)


print(
    f"Missing: "
    f"{score_data['missing']}/"
    f"{score_data['total']}"
)


print(
    f"Not Verifiable: "
    f"{score_data['not_verifiable']}/"
    f"{score_data['total']}"
)


# ============================================================
# REQUIREMENT EVALUATION
# ============================================================

print("\n------------------------------")
print("REQUIREMENT EVALUATION")
print("------------------------------")


for item in evaluation.requirements:

    print(
        f"\nSkill: {item.requirement}"
    )

    print(
        f"Status: {item.status}"
    )

    print(
        f"Evidence: {item.evidence}"
    )


# ============================================================
# CANDIDATE SUMMARY
# ============================================================

print("\n------------------------------")
print("CANDIDATE SUMMARY")
print("------------------------------")


print(
    evaluation.candidate_summary
)


# ============================================================
# STRENGTHS
# ============================================================

print("\n------------------------------")
print("STRENGTHS")
print("------------------------------")


for strength in evaluation.strengths:

    print(
        f"- {strength}"
    )


# ============================================================
# WEAKNESSES
# ============================================================

print("\n------------------------------")
print("WEAKNESSES")
print("------------------------------")


for weakness in evaluation.weaknesses:

    print(
        f"- {weakness}"
    )


# ============================================================
# HIRING RECOMMENDATION
# ============================================================

print("\n------------------------------")
print("HIRING RECOMMENDATION")
print("------------------------------")


print(
    score_data["recommendation"]
)