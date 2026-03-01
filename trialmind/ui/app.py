"""
TrialMind Streamlit User Interface.

Run with: streamlit run trialmind/ui/app.py
"""

import streamlit as st
import requests
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import LLM_MODEL

st.set_page_config(
    page_title="TrialMind — Clinical Trial Protocol Optimizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
st.markdown("""
<style>
.main-header { font-size: 2rem; font-weight: 700; color: #1a365d; }
.sub-header { font-size: 1rem; color: #4a5568; margin-bottom: 2rem; }
.metric-box { background: #f7fafc; border: 1px solid #e2e8f0;
              border-radius: 8px; padding: 1rem; }
.evidence-card { background: #fff; border-left: 4px solid #3182ce;
                 padding: 0.75rem; margin: 0.5rem 0; border-radius: 4px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.risk-high { color: #c53030; font-weight: 600; }
.risk-medium { color: #c05621; font-weight: 600; }
.risk-low { color: #276749; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔬 TrialMind</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Clinical Trial Protocol Optimizer — '
    'Evidence-Based Design from 450,000+ Historical Trials</div>',
    unsafe_allow_html=True
)

# ── Sidebar: Protocol Builder ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Protocol Context (Optional)")
    st.caption("Fill in your protocol details for a comprehensive review")

    indication = st.text_input(
        "Indication / Disease",
        placeholder="e.g., Non-small cell lung cancer"
    )
    phase = st.selectbox(
        "Trial Phase",
        ["", "Phase 1", "Phase 2", "Phase 3", "Phase 4",
         "Phase 1/Phase 2", "Phase 2/Phase 3"]
    )
    drug_name = st.text_input(
        "Drug / Intervention",
        placeholder="e.g., Pembrolizumab"
    )
    planned_n = st.number_input("Planned Sample Size (N)", min_value=0, value=0)
    primary_endpoint = st.text_input(
        "Primary Endpoint",
        placeholder="e.g., Progression-free survival"
    )
    duration = st.number_input("Planned Duration (months)", min_value=0, value=0)

    with st.expander("Eligibility Criteria"):
        inclusion_text = st.text_area(
            "Inclusion Criteria (one per line)",
            placeholder="ECOG PS 0-1\nHistologically confirmed NSCLC\nPD-L1 TPS ≥ 1%"
        )
        exclusion_text = st.text_area(
            "Exclusion Criteria (one per line)",
            placeholder="Prior immunotherapy\nActive autoimmune disease\nBrain metastases"
        )

    countries = st.multiselect(
        "Planned Countries",
        ["United States", "European Union", "United Kingdom", "Canada",
         "Japan", "China", "South Korea", "Australia", "India", "Brazil",
         "Germany", "France", "Italy", "Spain"]
    )

    dropout_pct = st.slider("Assumed Dropout Rate (%)", 0, 50, 15)

    # Build protocol context dict
    protocol_context = None
    if any([indication, phase, drug_name, planned_n > 0]):
        inclusion_list = [x.strip() for x in inclusion_text.split('\n') if x.strip()] if inclusion_text else []
        exclusion_list = [x.strip() for x in exclusion_text.split('\n') if x.strip()] if exclusion_text else []

        protocol_context = {
            "indication": indication or None,
            "phase": phase or None,
            "drug_name": drug_name or None,
            "planned_enrollment": planned_n if planned_n > 0 else None,
            "primary_endpoint": primary_endpoint or None,
            "duration_months": duration if duration > 0 else None,
            "inclusion_criteria": inclusion_list,
            "exclusion_criteria": exclusion_list,
            "countries": countries,
            "dropout_assumption": dropout_pct / 100 if dropout_pct > 0 else None
        }

    # API URL configuration
    st.markdown("---")
    api_url = st.text_input("API URL", value="http://localhost:8000")

# ── Main Interface ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Query Interface",
    "📋 Full Protocol Review",
    "📊 Benchmarks"
])

# ── TAB 1: Query Interface ────────────────────────────────────────────────────
with tab1:
    st.subheader("Ask TrialMind")

    # Quick query examples
    st.caption("Example queries:")

    example_queries = {
        "Sample Size": "What sample sizes are typical for Phase 2 NSCLC trials with PD-L1 biomarker selection?",
        "Endpoint": "What primary endpoints has FDA accepted for Phase 3 heart failure trials in the last 5 years?",
        "Eligibility": "What eligibility criteria in Phase 2 oncology trials most commonly cause recruitment failure?",
        "Site": "Which countries show the fastest enrollment rates for Phase 3 cardiovascular trials?",
        "Failure": "What are the most common reasons Phase 3 oncology trials are terminated early?",
        "Dropout": "What dropout rates should I model for a 24-month Phase 3 Alzheimer's trial?"
    }

    selected_example = None
    cols = st.columns(3)
    for idx, (label, query_text) in enumerate(example_queries.items()):
        with cols[idx % 3]:
            if st.button(f"📌 {label}", key=f"ex_{label}"):
                selected_example = query_text

    query = st.text_area(
        "Your question:",
        value=selected_example or "",
        height=120,
        placeholder=(
            "Ask about sample sizes, endpoints, eligibility, sites, "
            "enrollment rates, dropout, or trial failures..."
        )
    )

    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving evidence from 450,000+ clinical trials..."):
                try:
                    payload = {
                        "query": query,
                        "protocol": protocol_context
                    }
                    response = requests.post(
                        f"{api_url}/query",
                        json=payload,
                        timeout=120
                    )

                    if response.status_code == 200:
                        result = response.json()

                        # Metrics row
                        col1, col2, col3 = st.columns(3)
                        col1.metric(
                            "Query Type",
                            result['intent'].replace('_', ' ').title()
                        )
                        col2.metric("Trials Analyzed", result['trial_count_retrieved'])
                        col3.metric("NCT IDs Cited", len(result['nct_ids_referenced']))

                        # Main analysis output
                        st.markdown("---")
                        st.markdown("### Analysis")
                        st.markdown(result['analysis'])

                        # NCT IDs as clickable links
                        if result['nct_ids_referenced']:
                            st.markdown("---")
                            st.markdown("**Referenced Trials:**")
                            nct_links = " | ".join([
                                f"[{nct}](https://clinicaltrials.gov/study/{nct})"
                                for nct in result['nct_ids_referenced']
                            ])
                            st.markdown(nct_links)

                        # Download options
                        col1, col2 = st.columns(2)
                        col1.download_button(
                            "📄 Download Analysis (TXT)",
                            data=result['analysis'],
                            file_name="trialmind_analysis.txt",
                            mime="text/plain"
                        )
                        col2.download_button(
                            "📊 Download as JSON",
                            data=json.dumps(result, indent=2),
                            file_name="trialmind_result.json",
                            mime="application/json"
                        )
                    else:
                        st.error(f"API error: {response.status_code} — {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to TrialMind API. "
                        "Start the API with: `uvicorn trialmind.api.main:app --reload`"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

# ── TAB 2: Full Protocol Review ───────────────────────────────────────────────
with tab2:
    st.subheader("Comprehensive Protocol Review")
    st.caption(
        "Fill in the Protocol Context in the sidebar, then click Review. "
        "This generates a complete protocol optimization report with "
        "scorecard, risk assessment, and prioritized recommendations."
    )

    review_disabled = protocol_context is None

    if st.button(
        "🔬 Run Full Protocol Review",
        type="primary",
        use_container_width=True,
        disabled=review_disabled
    ):
        if not protocol_context:
            st.info("Please fill in the Protocol Context in the sidebar to run a full review.")
        else:
            with st.spinner("Running comprehensive protocol analysis (this may take 60-90 seconds)..."):
                try:
                    response = requests.post(
                        f"{api_url}/protocol-review",
                        json=protocol_context,
                        timeout=180
                    )
                    if response.status_code == 200:
                        result = response.json()
                        trial_count = len(result.get('retrieved_trials', []))
                        st.success(f"Analysis complete — {trial_count} comparable trials analyzed")
                        st.markdown(result['analysis'])
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    if review_disabled:
        st.info("👈 Fill in the Protocol Context in the sidebar to enable full review.")

# ── TAB 3: Benchmarks ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("Indication Benchmarks")

    col1, col2 = st.columns(2)
    with col1:
        bench_indication = st.text_input(
            "Indication",
            placeholder="e.g., Non-small cell lung cancer"
        )
    with col2:
        bench_phase = st.selectbox(
            "Phase",
            ["All phases", "Phase 1", "Phase 2", "Phase 3"],
            key="bench_phase"
        )

    if st.button("📊 Get Benchmarks", type="primary"):
        if bench_indication:
            with st.spinner(f"Generating benchmarks for {bench_indication}..."):
                phase_param = bench_phase if bench_phase != "All phases" else None
                try:
                    response = requests.get(
                        f"{api_url}/benchmark/{bench_indication}",
                        params={"phase": phase_param},
                        timeout=120
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.markdown(result['benchmark'])
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to TrialMind API. "
                        "Start the API with: `uvicorn trialmind.api.main:app --reload`"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter an indication.")

    # System health
    st.markdown("---")
    if st.button("🔄 Check System Status"):
        try:
            response = requests.get(f"{api_url}/health", timeout=10)
            if response.status_code == 200:
                health = response.json()
                st.success(f"System Status: {health['status']}")
                if health.get('collections'):
                    st.markdown("**Collection Sizes:**")
                    for name, count in health['collections'].items():
                        st.text(f"  {name}: {count:,} documents")
            else:
                st.error("API not responding")
        except Exception:
            st.warning("Cannot reach API — start with: `uvicorn trialmind.api.main:app --reload`")
