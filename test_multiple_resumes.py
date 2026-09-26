from rag_pipeline import screen_multiple_resumes


# ============================================================
# RESUMES
# ============================================================

PDF_PATHS = [
    "sample_resumes/Sumit_Naresh_Ghodke_Sep_2026_Resume.pdf"
]


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
# SCREEN MULTIPLE RESUMES
# ============================================================

results = screen_multiple_resumes(
    PDF_PATHS,
    JOB_DESCRIPTION
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("MULTIPLE RESUME SCREENING")
print("========================================")


for result in results:

    print("\n----------------------------------------")

    print(
        f"Resume: {result['pdf_path']}"
    )

    print("----------------------------------------")

    if result["success"]:

        evaluation = result["result"]["evaluation"]

        score_data = result["result"]["score_data"]


        print(
            f"Match Score: "
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


        print(
            f"Recommendation: "
            f"{score_data['recommendation']}"
        )


        print("\nCandidate Summary:")

        print(
            evaluation.candidate_summary
        )


    else:

        print(
            "ERROR:",
            result["error"]
        )