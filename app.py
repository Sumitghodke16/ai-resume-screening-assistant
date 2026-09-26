import shutil
from pathlib import Path

import streamlit as st

from rag_pipeline import screen_multiple_resumes


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background-color: #f8fafc;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    /* ======================================================
       HEADINGS
       ====================================================== */

    h1 {
        color: #0f172a !important;
        font-weight: 800 !important;
        letter-spacing: -0.035em;
    }

    h2 {
        color: #0f172a !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* ======================================================
       CAPTIONS
       ====================================================== */

    .stCaption {
        color: #64748b !important;
    }

    /* ======================================================
       TEXT AREA
       ====================================================== */

    .stTextArea textarea {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        color: #0f172a !important;
        font-size: 0.92rem !important;
        line-height: 1.55 !important;
    }

    .stTextArea textarea:focus {
        border-color: #64748b !important;
        box-shadow: 0 0 0 1px #64748b !important;
    }

    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    div[data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.25rem;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background-color: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 10px;
    }

    /* ======================================================
       BUTTON
       ====================================================== */

    div.stButton > button {
        min-height: 2.8rem;
        border-radius: 9px;
        font-weight: 700;
        border: none;
    }

    div.stButton > button:hover {
        border: none;
    }

    /* ======================================================
       METRICS
       ====================================================== */

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
    }

    /* ======================================================
       EXPANDERS
       ====================================================== */

    details {
        background-color: #ffffff;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
    }

    /* ======================================================
       DIVIDER
       ====================================================== */

    hr {
        border-color: #e2e8f0 !important;
    }

    /* ======================================================
       PROGRESS BAR
       ====================================================== */

    div[data-testid="stProgress"] > div {
        border-radius: 999px;
    }

    /* ======================================================
       ALERTS
       ====================================================== */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* ======================================================
       SIDEBAR CARDS
       ====================================================== */

    .sidebar-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.85rem;
        margin-bottom: 0.65rem;
    }

    .sidebar-card-title {
        color: #f8fafc;
        font-size: 0.84rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .sidebar-card-description {
        color: #94a3b8;
        font-size: 0.74rem;
        line-height: 1.45;
    }

    /* ======================================================
       CANDIDATE RESULT CONTAINER
       ====================================================== */

    .candidate-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* ======================================================
       SCORE DISPLAY
       ====================================================== */

    .score-display {
        text-align: center;
    }

    .score-number {
        color: #0f172a;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
    }

    .score-caption {
        color: #64748b;
        font-size: 0.7rem;
        font-weight: 700;
        margin-top: 0.3rem;
        letter-spacing: 0.04em;
    }

    /* ======================================================
       RECOMMENDATION BADGES
       ====================================================== */

    .strong-match {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 10px;
        color: #047857;
        font-weight: 750;
        padding: 0.7rem;
        text-align: center;
    }

    .moderate-match {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 10px;
        color: #a16207;
        font-weight: 750;
        padding: 0.7rem;
        text-align: center;
    }

    .weak-match {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 10px;
        color: #b91c1c;
        font-weight: 750;
        padding: 0.7rem;
        text-align: center;
    }

    /* ======================================================
       FOOTER
       ====================================================== */

    .footer-text {
        color: #94a3b8;
        font-size: 0.75rem;
        text-align: center;
        padding-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "screening_results" not in st.session_state:
    st.session_state.screening_results = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 📄 ResumeAI")



    st.markdown("### PIPELINE")

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">
                🔎 Retrieval
            </div>
            <div class="sidebar-card-description">
                FAISS vector search retrieves relevant
                resume evidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">
                🧠 Local AI
            </div>
            <div class="sidebar-card-description">
                Llama 3.2 runs locally through Ollama.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">
                📊 Evaluation
            </div>
            <div class="sidebar-card-description">
                Structured evaluation with deterministic
                match scoring.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### SUPPORTED INPUT")

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">
                Resume Format
            </div>
            <div class="sidebar-card-description">
                PDF files only
            </div>
        </div>

        <div class="sidebar-card">
            <div class="sidebar-card-title">
                Candidates
            </div>
            <div class="sidebar-card-description">
                Upload one or multiple resumes
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### SYSTEM")

    st.caption(
        "Local processing • Ollama + Llama 3.2"
    )


# ============================================================
# MAIN HEADER
# ============================================================


st.title(
    "AI Resume Screening Assistant"
)

st.write(
    "Evaluate candidate resumes against a job description "
    "using RAG, semantic retrieval, structured AI analysis, "
    "and deterministic match scoring."
)


# ============================================================
# INPUT SECTION
# ============================================================

left_column, right_column = st.columns(
    [1.15, 1],
    gap="large",
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

with left_column:

    st.subheader(
        "1. Job Description"
    )

    st.caption(
        "Enter the role requirements you want to evaluate "
        "candidates against."
    )

    default_jd = """We are looking for a Data Scientist.

Required skills:
- Python
- SQL
- Machine Learning
- Statistics
- Scikit-learn
- Predictive Modeling
- Data Analysis

The candidate should have experience building
machine learning models and analyzing business data."""

    job_description = st.text_area(
        "Job Description",
        value=default_jd,
        height=300,
        label_visibility="collapsed",
        placeholder="Paste the job description here...",
    )


# ============================================================
# RESUME UPLOAD
# ============================================================

with right_column:

    st.subheader(
        "2. Candidate Resumes"
    )

    st.caption(
        "Upload one or more candidate resumes in PDF format."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF resumes",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="You can upload multiple PDF resumes.",
    )

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} resume"
            f"{'s' if len(uploaded_files) != 1 else ''} "
            "ready for screening."
        )

        for uploaded_file in uploaded_files:

            size_mb = uploaded_file.size / (
                1024 * 1024
            )

            st.caption(
                f"📄 {uploaded_file.name} "
                f"• {size_mb:.2f} MB"
            )

    else:

        st.info(
            "Upload at least one PDF resume to begin."
        )


# ============================================================
# SCREEN BUTTON
# ============================================================

st.write("")

screen_button = st.button(
    "🚀  Screen Resumes",
    type="primary",
    use_container_width=True,
)


# ============================================================
# SCREEN RESUMES
# ============================================================

if screen_button:

    # --------------------------------------------------------
    # Validate Job Description
    # --------------------------------------------------------

    if not job_description.strip():

        st.error(
            "Please enter a job description before screening."
        )

        st.stop()


    # --------------------------------------------------------
    # Validate Resumes
    # --------------------------------------------------------

    if not uploaded_files:

        st.error(
            "Please upload at least one PDF resume."
        )

        st.stop()


    # --------------------------------------------------------
    # Temporary Directory
    # --------------------------------------------------------

    temp_dir = Path(
        ".streamlit_temp_resumes"
    )

    temp_dir.mkdir(
        exist_ok=True
    )


    pdf_paths = []


    # --------------------------------------------------------
    # Save Uploaded Files
    # --------------------------------------------------------

    for index, uploaded_file in enumerate(
        uploaded_files
    ):

        safe_filename = Path(
            uploaded_file.name
        ).name

        temp_path = (
            temp_dir
            / f"{index}_{safe_filename}"
        )

        with open(
            temp_path,
            "wb"
        ) as output_file:

            output_file.write(
                uploaded_file.getbuffer()
            )

        pdf_paths.append(
            str(temp_path)
        )


    # --------------------------------------------------------
    # Run Screening
    # --------------------------------------------------------

    progress_text = st.empty()

    progress_bar = st.progress(
        0
    )

    progress_text.info(
        "🔎 Preparing resumes..."
    )

    try:

        progress_bar.progress(
            20
        )

        progress_text.info(
            "🧠 Running RAG-based AI screening..."
        )

        results = screen_multiple_resumes(
            pdf_paths,
            job_description,
        )

        progress_bar.progress(
            100
        )

        progress_text.success(
            "✓ Screening completed successfully."
        )

        st.session_state.screening_results = (
            results
        )

    except Exception as error:

        progress_bar.empty()
        progress_text.empty()

        st.error(
            "The screening process encountered an error."
        )

        with st.expander(
            "Technical details"
        ):

            st.code(
                str(error)
            )

        st.stop()

    finally:

        # ----------------------------------------------------
        # Remove temporary uploaded files
        # ----------------------------------------------------

        try:

            if temp_dir.exists():

                shutil.rmtree(
                    temp_dir
                )

        except Exception:

            pass


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.screening_results


if results:

    st.divider()

    st.header(
        "Screening Results"
    )

    st.caption(
        "Review match scores, required skills, evidence, "
        "strengths, weaknesses, and recommendations."
    )


    # ========================================================
    # SEPARATE SUCCESS / FAILURE
    # ========================================================

    successful_results = [
        result
        for result in results
        if result.get("success")
    ]

    failed_results = [
        result
        for result in results
        if not result.get("success")
    ]


    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    if successful_results:

        scores = [
            result["result"]["score_data"]["score"]
            for result in successful_results
        ]

        average_score = round(
            sum(scores) / len(scores)
        )

        strong_matches = sum(
            1
            for result in successful_results
            if result["result"]["score_data"][
                "recommendation"
            ] == "Strong Match"
        )


        metric_1, metric_2, metric_3, metric_4 = (
            st.columns(4)
        )


        with metric_1:

            st.metric(
                "Resumes Screened",
                len(results),
            )


        with metric_2:

            st.metric(
                "Average Match",
                f"{average_score}/100",
            )


        with metric_3:

            st.metric(
                "Strong Matches",
                strong_matches,
            )


        with metric_4:

            st.metric(
                "Processing Errors",
                len(failed_results),
            )


    # ========================================================
    # CANDIDATE RESULTS
    # ========================================================

    for index, item in enumerate(
        results
    ):

        filename = Path(
            item["pdf_path"]
        ).name


        # ====================================================
        # FAILED RESUME
        # ====================================================

        if not item.get("success"):

            with st.expander(
                f"⚠️ Candidate {index + 1} — {filename}"
            ):

                st.error(
                    item.get(
                        "error",
                        "Unknown processing error.",
                    )
                )

            continue


        # ====================================================
        # SUCCESSFUL RESUME
        # ====================================================

        result = item["result"]

        evaluation = result["evaluation"]

        score_data = result["score_data"]

        score = score_data["score"]

        recommendation = (
            score_data["recommendation"]
        )


        # ----------------------------------------------------
        # Candidate container
        # ----------------------------------------------------

        with st.container(
            border=True
        ):

            # ------------------------------------------------
            # Candidate header
            # ------------------------------------------------

            header_left, header_score, header_status = (
                st.columns(
                    [2.5, 0.8, 1.3],
                    gap="large",
                )
            )


            with header_left:

                st.subheader(
                    f"👤 Candidate {index + 1}"
                )

                st.caption(
                    f"📄 {filename}"
                )


            with header_score:

                st.markdown(
                    f"""
                    <div class="score-display">
                        <div class="score-number">
                            {score}
                        </div>
                        <div class="score-caption">
                            MATCH SCORE / 100
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            with header_status:

                if recommendation == "Strong Match":

                    recommendation_class = (
                        "strong-match"
                    )

                elif recommendation == "Moderate Match":

                    recommendation_class = (
                        "moderate-match"
                    )

                else:

                    recommendation_class = (
                        "weak-match"
                    )

                st.markdown(
                    f"""
                    <div class="{recommendation_class}">
                        {recommendation}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            # ------------------------------------------------
            # Score bar
            # ------------------------------------------------

            st.progress(
                score / 100
            )


            # ------------------------------------------------
            # Evaluation metrics
            # ------------------------------------------------

            metric_a, metric_b, metric_c = (
                st.columns(3)
            )


            with metric_a:

                st.metric(
                    "Matching",
                    score_data["matching"],
                )


            with metric_b:

                st.metric(
                    "Missing",
                    score_data["missing"],
                )


            with metric_c:

                st.metric(
                    "Not Verifiable",
                    score_data["not_verifiable"],
                )


            # ------------------------------------------------
            # Candidate Summary
            # ------------------------------------------------

            st.markdown(
                "#### Candidate Summary"
            )

            st.write(
                evaluation.candidate_summary
            )


            # ------------------------------------------------
            # Requirement Evaluation
            # ------------------------------------------------

            st.markdown(
                "#### Requirement Evaluation"
            )


            matching_skills = []

            missing_skills = []

            unverifiable_skills = []


            for requirement in (
                evaluation.requirements
            ):

                if requirement.status == "MATCHING":

                    matching_skills.append(
                        requirement.requirement
                    )

                elif requirement.status == "MISSING":

                    missing_skills.append(
                        requirement.requirement
                    )

                else:

                    unverifiable_skills.append(
                        requirement.requirement
                    )


            # ------------------------------------------------
            # Matching skills
            # ------------------------------------------------

            if matching_skills:

                st.markdown(
                    "**Matching Skills**"
                )

                st.success(
                    " • ".join(
                        matching_skills
                    )
                )


            # ------------------------------------------------
            # Missing skills
            # ------------------------------------------------

            if missing_skills:

                st.markdown(
                    "**Missing Skills**"
                )

                st.error(
                    " • ".join(
                        missing_skills
                    )
                )


            # ------------------------------------------------
            # Not verifiable
            # ------------------------------------------------

            if unverifiable_skills:

                st.markdown(
                    "**Not Verifiable**"
                )

                st.warning(
                    " • ".join(
                        unverifiable_skills
                    )
                )


            # ------------------------------------------------
            # Detailed Evidence
            # ------------------------------------------------

            with st.expander(
                "🔍 View requirement evidence"
            ):

                for requirement in (
                    evaluation.requirements
                ):

                    st.markdown(
                        f"**{requirement.requirement}**"
                    )

                    st.caption(
                        f"Status: {requirement.status}"
                    )

                    st.write(
                        requirement.evidence
                    )

                    st.divider()


            # ------------------------------------------------
            # Strengths and Weaknesses
            # ------------------------------------------------

            strength_column, weakness_column = (
                st.columns(
                    2,
                    gap="large",
                )
            )


            with strength_column:

                st.markdown(
                    "#### Strengths"
                )

                if evaluation.strengths:

                    for strength in (
                        evaluation.strengths
                    ):

                        st.markdown(
                            f"✓ {strength}"
                        )

                else:

                    st.caption(
                        "No specific strengths identified."
                    )


            with weakness_column:

                st.markdown(
                    "#### Weaknesses"
                )

                if evaluation.weaknesses:

                    for weakness in (
                        evaluation.weaknesses
                    ):

                        st.markdown(
                            f"• {weakness}"
                        )

                else:

                    st.caption(
                        "No specific weaknesses identified."
                    )


else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.divider()

    st.info(
        "📄  Ready to screen candidates"
    )

    st.caption(
        "Enter a job description, upload one or more "
        "PDF resumes, and click 'Screen Resumes' to begin."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Resume Screening Assistant • "
    "RAG + FAISS + Llama 3.2 • Local AI Processing"
)