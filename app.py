"""
Patent Novelty Assessment System - Streamlit UI

A clean, minimal interface for assessing patent novelty and searching prior art.

Run with: streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path
import json
import time
from dataclasses import asdict

_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

st.set_page_config(
    page_title="Patent Novelty Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

@st.cache_resource
def load_analyzer(serpapi_key: str = None, use_online: bool = True, use_keywords: bool = True):
    """Load the patent analyzer with Hybrid RAG settings."""
    from src.app.patent_analyzer import PatentAnalyzer
    
    return PatentAnalyzer(
        use_online_search=use_online,
        use_llm_keywords=use_keywords,
        serpapi_key=serpapi_key
    )


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>Patent Novelty Assessment System</h1>
        <p>Analyze patent novelty and search prior art using hybrid retrieval and ML classification</p>
    </div>
    """, unsafe_allow_html=True)


def _summarize_abstract(abstract_text: str, max_sentences: int = 4) -> str:
    """Summarize abstract to first few sentences."""
    if not abstract_text or abstract_text == 'No abstract available':
        return abstract_text
    
    sentences = [s.strip() for s in abstract_text.replace('\n', ' ').split('.') if s.strip()]
    if not sentences:
        trimmed = abstract_text.strip()
        if len(trimmed) > 500:
            trimmed = trimmed[:500].rsplit(' ', 1)[0]
        return trimmed
    
    selected = sentences[:max_sentences]
    summary = '. '.join(selected)
    if len(summary) > 500:
        summary = summary[:500].rsplit(' ', 1)[0]
    
    return summary.strip()


def _extract_key_claims(claims: list, max_claims: int = 3) -> list:
    """Extract key claims (prefer independent claims, limit to max_claims)."""
    if not claims:
        return []
    
    # Separate independent and dependent claims
    independent = []
    dependent = []
    
    for claim in claims:
        if isinstance(claim, dict):
            is_dependent = claim.get('dependent', False)
            claim_text = claim.get('text', '')
            claim_num = claim.get('claim_num', 0)
            
            if not is_dependent and claim_text:
                independent.append((claim_num, claim_text))
            elif claim_text:
                dependent.append((claim_num, claim_text))
        else:
            # Fallback: treat as independent
            independent.append((0, str(claim)))
    
    # Prefer independent claims, fall back to dependent if needed
    key_claims = []
    if independent:
        # Sort by claim number and take first max_claims
        independent.sort(key=lambda x: x[0])
        key_claims = [{'text': text, 'claim_num': num} for num, text in independent[:max_claims]]
    elif dependent:
        # If no independent claims, use dependent
        dependent.sort(key=lambda x: x[0])
        key_claims = [{'text': text, 'claim_num': num} for num, text in dependent[:max_claims]]
    
    return key_claims


def _find_overlapping_quotes(query_text: str, patent_text: str, max_quotes: int = 3) -> list:
    """Find overlapping phrases between query and patent text."""
    if not query_text or not patent_text:
        return []
    
    import re
    
    # Extract meaningful words (3+ characters) from query
    query_words = re.findall(r'\b\w{3,}\b', query_text.lower())
    if len(query_words) < 2:
        return []
    
    # Find phrases in patent text that contain query words
    patent_lower = patent_text.lower()
    overlapping_phrases = []
    
    # Look for 3-5 word phrases containing query terms
    for i in range(len(query_words) - 1):
        # Try 3-word, 4-word, 5-word phrases
        for phrase_len in [3, 4, 5]:
            if i + phrase_len <= len(query_words):
                phrase = ' '.join(query_words[i:i+phrase_len])
                # Find this phrase in patent text
                if phrase in patent_lower:
                    # Extract context around the match
                    idx = patent_lower.find(phrase)
                    start = max(0, idx - 50)
                    end = min(len(patent_text), idx + len(phrase) + 50)
                    context = patent_text[start:end].strip()
                    if context and context not in overlapping_phrases:
                        overlapping_phrases.append(context)
    
    # Return top max_quotes
    return overlapping_phrases[:max_quotes]


def render_prior_art(patents: list, max_display: int = 10, context: str = "default", query_text: str = ""):
    import html
    
    if not patents:
        st.info("No similar patents found.")
        return
    
    sort_key = f"sort_patents_{context}"
    count_key = f"show_count_{context}"
    filter_key = f"filter_source_{context}"
    
    col1, col2 = st.columns([2, 1])
    with col1:
        sort_index = 0
        if sort_key in st.session_state:
            sort_val = st.session_state[sort_key]
            if isinstance(sort_val, int):
                sort_index = sort_val
            elif isinstance(sort_val, str):
                options = ["Similarity (High to Low)", "Similarity (Low to High)", "Year (Recent First)", "Year (Oldest First)"]
                sort_index = options.index(sort_val) if sort_val in options else 0
        
        sort_by = st.selectbox(
            "Sort by:",
            ["Similarity (High to Low)", "Similarity (Low to High)", "Year (Recent First)", "Year (Oldest First)"],
            key=sort_key,
            index=sort_index
        )
    with col2:
        default_count = max_display
        if count_key in st.session_state:
            count_val = st.session_state[count_key]
            if isinstance(count_val, (int, float)):
                default_count = int(count_val)
        
        show_count = st.slider(
            "Show patents:", 
            5, 
            min(50, len(patents)), 
            default_count, 
            key=count_key
        )
    
    sorted_patents = patents.copy()
    if "Similarity" in sort_by:
        # Use similarity field for sorting
        sorted_patents.sort(key=lambda x: x.get('similarity', 0) or x.get('similarity_score', 0), 
                          reverse="High" in sort_by)
    elif "Year" in sort_by:
        sorted_patents.sort(key=lambda x: int(x.get('year', 0)) if str(x.get('year', 'N/A')).isdigit() else 0,
                          reverse="Recent" in sort_by)
    
    if any(p.get('source') for p in patents):
        default_filter = st.session_state.get(filter_key, 0)
        if isinstance(default_filter, str):
            default_filter = ["All", "Local Database", "Online Search"].index(default_filter) if default_filter in ["All", "Local Database", "Online Search"] else 0
        elif not isinstance(default_filter, int):
            default_filter = 0
        
        filter_source = st.radio(
            "Filter by source:",
            ["All", "Local Database", "Online Search"],
            horizontal=True,
            key=filter_key,
            index=int(default_filter)
        )
        if filter_source == "Local Database":
            sorted_patents = [p for p in sorted_patents if p.get('source') != 'online' and not ('google' in str(p.get('patent_id', '')).lower() or any(x in str(p.get('patent_id', '')) for x in ['CN', 'JP', 'EP', 'WO']))]
        elif filter_source == "Online Search":
            sorted_patents = [p for p in sorted_patents if p.get('source') == 'online' or 'google' in str(p.get('patent_id', '')).lower() or any(x in str(p.get('patent_id', '')) for x in ['CN', 'JP', 'EP', 'WO'])]
    
    st.markdown(f"**Showing {min(show_count, len(sorted_patents))} of {len(patents)} patents**")
    
    for i, patent in enumerate(sorted_patents[:show_count]):
        patent_id = patent.get('patent_id', 'N/A')
        title_raw = patent.get('title', 'No title available')
        abstract_raw = patent.get('abstract', 'No abstract available')
        title = html.escape(str(title_raw) if title_raw and not isinstance(title_raw, dict) else 'No title available')
        
        # Process abstract: ensure good coverage and ends with period
        abstract_text = str(abstract_raw) if abstract_raw and not isinstance(abstract_raw, dict) else 'No abstract available'
        abstract_summary = _summarize_abstract(abstract_text, max_sentences=4)
        abstract = html.escape(abstract_summary)
        
        year = patent.get('year', 'N/A')
        if year == 'N/A' or not year or year == 0:
            grant_date = patent.get('grant_date', '') or patent.get('publication_date', '')
            if grant_date:
                year = str(grant_date)[:4] if len(str(grant_date)) >= 4 else 'N/A'
            else:
                year = 'N/A'
        else:
            year = str(year) if isinstance(year, (int, float)) else year
        # Get similarity score (prefer model_similarity if available)
        similarity = patent.get('model_similarity')
        if similarity is None:
            similarity = patent.get('similarity', patent.get('similarity_score', 0.0))
        model_novelty = patent.get('model_novelty')
        rank = patent.get('rank')
        
        source_badge = ""
        if patent.get('source') == 'online' or 'google' in str(patent_id).lower() or any(x in str(patent_id) for x in ['CN', 'JP', 'EP', 'WO']):
            source_badge = " [Online]"
        else:
            source_badge = " [Local]"
        
        # Build title with rank if available
        title_prefix = f"#{rank} - " if rank else ""
        score_suffix = f" - Similarity: {similarity:.3f}"
        if model_novelty is not None:
            score_suffix = f" - Model Similarity: {similarity:.3f} | Novelty: {model_novelty:.3f}"
        
        with st.expander(f"**{title_prefix}{title}** (Patent {patent_id}, {year}){score_suffix}{source_badge}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if rank:
                    st.markdown(f"**Rank:** #{rank}")
                st.markdown(f"**Patent ID:** {patent_id}")
                st.markdown(f"**Year:** {year}")
                st.markdown(f"**Similarity:** {similarity:.3f}")
                if model_novelty is not None:
                    st.markdown(f"**Model Novelty:** {model_novelty:.3f}")
            with col2:
                if patent.get('cpc_code'):
                    st.markdown(f"**CPC Code:** {patent.get('cpc_code')}")
                if patent.get('assignee'):
                    st.markdown(f"**Assignee:** {patent.get('assignee')}")
                if patent.get('inventor'):
                    st.markdown(f"**Inventor:** {patent.get('inventor')}")
            
            # Filter out SerpAPI URLs
            patent_url = None
            if patent.get('url') or patent.get('link'):
                url = patent.get('url') or patent.get('link')
                # Only show URL if it's not a SerpAPI link
                if url and 'serpapi.com' not in str(url).lower():
                    patent_url = url
            
            # Summarize abstract (longer, no claims)
            abstract_summary = _summarize_abstract(abstract_text, max_sentences=4)
            st.markdown(f"**Summary:** {abstract_summary}")
            
            # Show patent URL if available (and not SerpAPI)
            if patent_url:
                st.markdown(f"**Link:** [{patent_url}]({patent_url})")


def generate_report_text(result, input_text: str = ""):
    lines = []
    lines.append("=" * 80)
    lines.append("PATENT NOVELTY ASSESSMENT REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    if input_text:
        lines.append("INPUT PATENT:")
        lines.append("-" * 80)
        lines.append(input_text[:500] + ("..." if len(input_text) > 500 else ""))
        lines.append("")
    
    score = result.novelty_score if result.novelty_score is not None else 0
    lines.append("NOVELTY ASSESSMENT:")
    lines.append("-" * 80)
    lines.append(f"Novelty Score: {score:.4f}")
    # Remove hard categories - use continuous score with context
    assessment = result.assessment if result.assessment else None
    if assessment:
        lines.append(f"Novelty Score: {score:.3f} ({assessment})")
    else:
        lines.append(f"Novelty Score: {score:.3f}")
    
    if result.recommendation:
        lines.append(f"Recommendation: {result.recommendation}")
    lines.append("")
    
    if result.explanation:
        lines.append("DETAILED EXPLANATION:")
        lines.append("-" * 80)
        lines.append(result.explanation)
        lines.append("")
    
    similar_patents = result.similar_patents if result.similar_patents else []
    if similar_patents:
        lines.append(f"SIMILAR PATENTS FOUND ({len(similar_patents)} total):")
        lines.append("-" * 80)
        for i, patent in enumerate(similar_patents, 1):
            lines.append(f"\n{i}. Patent ID: {patent.get('patent_id', 'N/A')}")
            lines.append(f"   Similarity Score: {patent.get('similarity_score', patent.get('similarity', 0)):.4f}")
            lines.append(f"   Year: {patent.get('year', 'N/A')}")
            if patent.get('title'):
                lines.append(f"   Title: {patent.get('title', 'N/A')}")
            if patent.get('abstract'):
                abstract = str(patent.get('abstract', ''))[:300]
                lines.append(f"   Abstract: {abstract}...")
            lines.append("")
    
    if result.search_metadata:
        lines.append("SEARCH METADATA:")
        lines.append("-" * 80)
        for key, value in result.search_metadata.items():
            lines.append(f"{key}: {value}")
        lines.append("")
    
    lines.append("=" * 80)
    lines.append(f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def render_novelty_result(result, input_text: str = ""):
    score = result.novelty_score if result.novelty_score is not None else 0
    explanation = result.explanation if result.explanation else 'No explanation available.'
    similar_patents = result.similar_patents if result.similar_patents else []
    recommendation = result.recommendation if result.recommendation else ''
    assessment = result.assessment if result.assessment else None
    
    # Get ranking metadata if available
    search_metadata = result.search_metadata if result.search_metadata else {}
    rank_percentile = search_metadata.get('rank_percentile')
    top_k_scored = search_metadata.get('top_k_scored', 0)
    
    # Calculate similarity percentage
    similarity_pct = (1 - score) * 100
    
    # Determine assessment styling
    if score > 0.7:
        assessment_color = "#4CAF50"
        assessment_label = "HIGHLY NOVEL"
    elif score > 0.5:
        assessment_color = "#FFA726"
        assessment_label = "MODERATELY NOVEL"
    elif score > 0.3:
        assessment_color = "#FF9800"
        assessment_label = "LOW NOVELTY"
    else:
        assessment_color = "#EF5350"
        assessment_label = "NOT NOVEL"
    
    # Main assessment card
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%); border: 2px solid {assessment_color}; border-radius: 12px; padding: 2rem; margin: 1.5rem 0; text-align: center;">
        <div style="font-size: 3.5rem; font-weight: 700; color: {assessment_color}; margin-bottom: 0.5rem;">{score:.2f}</div>
        <div style="font-size: 1.5rem; font-weight: 600; color: {assessment_color}; margin-bottom: 1rem;">
            {assessment_label}
        </div>
        <div style="font-size: 1rem; color: #aaa; margin-top: 1rem;">
            {similarity_pct:.0f}% similar to prior art found
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Novelty Score", f"{score:.2f}", help="0.0 = Not Novel, 1.0 = Highly Novel")
    with col2:
        st.metric("Similar Patents", len(similar_patents), help="Number of similar prior art patents found")
    with col3:
        online_count = sum(1 for p in similar_patents if p.get('source') == 'online' or 'google' in str(p.get('patent_id', '')).lower())
        local_count = len(similar_patents) - online_count
        st.metric("Search Coverage", f"{local_count}L + {online_count}O", help="Local (L) and Online (O) patents found")
    with col4:
        if rank_percentile is not None and top_k_scored > 0:
            # Fix percentile display: rank_percentile represents % that are MORE novel
            # So if rank_percentile = 100%, it means 100% are more novel = this is least novel (0% novel)
            # We want to show it as "Top 0%" or "Bottom 100%"
            novelty_percentile = 100 - rank_percentile if rank_percentile else None
            if novelty_percentile is not None:
                st.metric("Novelty Rank", f"Top {novelty_percentile:.0f}%", help=f"Compared to {top_k_scored} analyzed patents")
    
    # Explanation section
    with st.expander("📊 What do these metrics mean?", expanded=False):
        st.markdown("""
        **Novelty Score (0.0 - 1.0)**
        - **0.0 - 0.3**: Not Novel - Significant overlap with existing patents
        - **0.3 - 0.5**: Low Novelty - Some unique aspects but substantial prior art
        - **0.5 - 0.7**: Moderately Novel - Notable differences from prior art
        - **0.7 - 1.0**: Highly Novel - Minimal similarity to existing patents
        
        **Similarity Percentage**
        - Shows how similar your patent is to the most similar prior art found
        - Lower percentage = more novel
        - Based on PyTorch neural network analysis of 10 engineered features
        
        **Novelty Rank**
        - Your patent's novelty compared to the top 20 most similar patents analyzed
        - "Top X%" means your patent is more novel than X% of similar patents
        - Based on ranking distribution of model similarity scores
        
        **Search Coverage**
        - **Local (L)**: Patents from our 200K patent database (2021-2025)
        - **Online (O)**: Patents found via Google Patents search (millions of patents)
        """)
    
    # Recommendation
    if recommendation:
        if score < 0.3:
            st.error(f"**Recommendation:** {recommendation}")
        elif score < 0.7:
            st.warning(f"**Recommendation:** {recommendation}")
        else:
            st.success(f"**Recommendation:** {recommendation}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        report_text = generate_report_text(result, input_text)
        st.download_button(
            label="Download Report (.txt)",
            data=report_text,
            file_name=f"patent_analysis_{int(time.time())}.txt",
            mime="text/plain"
        )
    with col2:
        report_data = {
            "novelty_score": score,
            "recommendation": recommendation,
            "explanation": explanation,
            "similar_patents": similar_patents,
            "search_metadata": result.search_metadata or {}
        }
        if assessment:
            report_data["assessment"] = assessment
        report_json = json.dumps(report_data, indent=2)
        st.download_button(
            label="Download JSON",
            data=report_json,
            file_name=f"patent_analysis_{int(time.time())}.json",
            mime="application/json"
        )
    with col3:
        if st.button("Copy to Clipboard"):
            st.code(report_text[:500] + "...", language="text")
            st.info("Full report available for download. Clipboard copy limited to 500 chars.")
    
    with st.expander("Detailed Explanation", expanded=True):
        st.markdown(explanation)


def main():
    """Main application."""
    if 'css_loaded' not in st.session_state:
        st.markdown("""
<style>
.stApp{background:#1a1a1a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
*{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important}
.main .block-container{padding-top:2rem;max-width:1200px}
.main-header{border-bottom:1px solid #333;padding-bottom:1rem;margin-bottom:2rem}
.main-header h1{color:#fff;font-size:1.75rem;font-weight:600;margin-bottom:.25rem}
.main-header p{color:#888;font-size:.9rem}
.score-box{background:#252525;border:1px solid #333;border-radius:8px;padding:1.5rem;text-align:center;margin:1rem 0}
.score-value{font-size:3rem;font-weight:600}
.score-high{color:#fff}
.score-medium{color:#aaa}
.score-low{color:#666}
.score-label{color:#888;font-size:.875rem;margin-top:.5rem}
section[data-testid="stSidebar"]{background:#141414;border-right:1px solid #2a2a2a}
section[data-testid="stSidebar"] .stMarkdown{color:#e0e0e0}
.stButton>button{background:#2a2a2a;color:#e0e0e0;border:1px solid #404040;border-radius:6px;padding:.5rem 1rem;font-weight:500;transition:all .2s}
.stButton>button:hover{background:#3a3a3a;border-color:#555;color:#fff}
.stButton>button[kind="primary"]{background:#fff;border-color:#fff;color:#1a1a1a}
.stButton>button[kind="primary"]:hover{background:#e0e0e0;border-color:#e0e0e0}
.stTextInput>div>div>input,.stTextArea>div>div>textarea{background:#252525;border:1px solid #333;border-radius:6px;color:#e0e0e0}
.stTextInput>div>div>input:focus,.stTextArea>div>div>textarea:focus{border-color:#666;box-shadow:none}
.stTabs [data-baseweb="tab-list"]{gap:0;background:transparent;border-bottom:1px solid #444}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:0;color:#8e8e8e;padding:.75rem 1.25rem;border-bottom:2px solid transparent}
.stTabs [aria-selected="true"]{background:transparent;color:#fff;border-bottom:2px solid #888}
.stTabs [data-baseweb="tab-highlight"]{background:#888!important}
.stTabs [data-baseweb="tab-border"]{background:transparent!important}
.streamlit-expanderHeader{background:#252525;border:1px solid #333;border-radius:6px;color:#e0e0e0;padding-left:1rem!important}
.streamlit-expanderHeader svg,.streamlit-expanderHeader svg*,button[data-testid*="expander"] svg,button[data-testid*="expander"] svg*,[data-testid*="expander"] svg,[data-testid*="expander"] svg*{display:none!important;visibility:hidden!important;opacity:0!important;width:0!important;height:0!important;margin:0!important;padding:0!important}
.streamlit-expanderHeader::before,.streamlit-expanderHeader::after,button[data-testid*="expander"]::before,button[data-testid*="expander"]::after,[data-testid*="expander"]::before,[data-testid*="expander"]::after{content:""!important;display:none!important}
button[data-testid*="expander"]{text-indent:0!important}
.material-icons:contains("keyboard_double_arrow_right"),*:contains("keyboard_double_arrow_right"){display:none!important;font-size:0!important}
.streamlit-expanderContent{background:#252525;border:1px solid #333;border-top:none;border-radius:0 0 6px 6px}
.stProgress>div>div{background:#888}
[data-testid="stMetricValue"]{color:#fff}
[data-testid="stMetricLabel"]{color:#888}
.stRadio>div{background:transparent}
.stRadio label{color:#e0e0e0!important}
.stSelectbox>div>div{background:#252525;border-color:#333}
.stSuccess{background:rgba(255,255,255,.05);border:1px solid #555}
.stWarning{background:rgba(255,255,255,.05);border:1px solid #444}
.stError{background:rgba(255,255,255,.05);border:1px solid #333}
#MainMenu{visibility:hidden}
footer{visibility:hidden}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:#1a1a1a}
::-webkit-scrollbar-thumb{background:#444;border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:#555}
h1,h2,h3,h4,h5,h6{color:#fff}
p,span,div,label{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
code{font-family:'SF Mono','Fira Code','Monaco',monospace!important;background:#2a2a2a;color:#e0e0e0}
a{color:#888}
a:hover{color:#fff}
</style>
""", unsafe_allow_html=True)
        st.session_state['css_loaded'] = True
    
    if 'analyzer_preloaded' not in st.session_state:
        st.session_state['analyzer_preloaded'] = False
    
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Configuration")
        
        serpapi_key = st.text_input(
            "SerpAPI Key (for online search)",
            value=st.session_state.get('serpapi_key', ''),
            type="password",
            help="Enter your SerpAPI key to enable online patent search via Google Patents"
        )
        
        if serpapi_key != st.session_state.get('serpapi_key', ''):
            st.session_state['serpapi_key'] = serpapi_key
            if serpapi_key:
                import os
                os.environ['SERPAPI_KEY'] = serpapi_key
                st.success(f"SerpAPI key configured ({len(serpapi_key)} characters)")
                load_analyzer.clear()
            else:
                st.info("SerpAPI key removed")
                load_analyzer.clear()
        
        if st.session_state.get('serpapi_key'):
            st.info(f"✓ SerpAPI key is set ({len(st.session_state.get('serpapi_key', ''))} characters)")
        
        use_online = st.checkbox(
            "Enable Online Search",
            value=st.session_state.get('use_online', True),
            help="Search Google Patents via SerpAPI for additional results"
        )
        st.session_state['use_online'] = use_online
        
        use_keywords = st.checkbox(
            "Use LLM Keyword Generation",
            value=st.session_state.get('use_keywords', True),
            help="Use Phi-3 to generate search keywords from input text"
        )
        st.session_state['use_keywords'] = use_keywords
        
        if st.button("Clear Cache"):
            load_analyzer.clear()
            st.session_state['analyzer_preloaded'] = False
            st.success("Cache cleared!")
    
    tab1, tab2 = st.tabs(["Novelty Assessment & Prior Art Search", "Quick Search"])
    
    with tab1:
        st.markdown("### Enter Patent Information")
        
        idea_text = st.text_area(
            "Describe your invention or paste patent text:",
            height=200,
            value=st.session_state.get('analysis_input', ''),
            placeholder="Enter a detailed description of your invention, including technical details, claims, and abstract..."
        )
        
        if st.button("Analyze Novelty", type="primary"):
            if not idea_text.strip():
                st.error("Please enter some text to analyze.")
            else:
                log_messages = []
                log_expander = st.expander("Analysis Log", expanded=False)
                
                with log_expander:
                    log_placeholder = st.empty()
                
                def analyze_and_display(input_data, is_search=False):
                    def update_status(message: str):
                        log_messages.append(message)
                        log_placeholder.markdown(
                            '<div style="max-height: 300px; overflow-y: auto; font-family: monospace; font-size: 0.85rem; background: #252525; padding: 1rem; border-radius: 4px;">' +
                            '<br>'.join([f'<div style="margin: 0.25rem 0; color: #e0e0e0;">{msg}</div>' for msg in log_messages]) +
                            '</div>',
                            unsafe_allow_html=True
                        )
                    
                    serpapi_key = st.session_state.get('serpapi_key', '')
                    use_online = st.session_state.get('use_online', True)
                    use_keywords = st.session_state.get('use_keywords', True)
                    
                    update_status("Loading analyzer...")
                    analyzer = load_analyzer(
                        serpapi_key=serpapi_key if serpapi_key else None,
                        use_online=use_online,
                        use_keywords=use_keywords
                    )
                    
                    update_status("Starting analysis...")
                    result = analyzer.analyze(input_data, status_callback=update_status)
                    
                    update_status("Analysis complete!")
                    
                    return result
                    
                try:
                    result = analyze_and_display(idea_text)
                    if result:
                        if result.success:
                            similar_count = len(result.similar_patents) if result.similar_patents else 0
                            print(f"DEBUG: Storing result with {similar_count} similar patents")
                            print(f"DEBUG: Result type: {type(result)}")
                            print(f"DEBUG: Result has similar_patents: {hasattr(result, 'similar_patents')}")
                            
                            # Store result in session state - ensure it's properly stored
                            try:
                                st.session_state['analysis_result'] = result
                                st.session_state['analysis_input'] = idea_text
                                print(f"DEBUG: Result stored successfully")
                                
                                # Verify it was stored
                                if 'analysis_result' in st.session_state:
                                    stored = st.session_state['analysis_result']
                                    print(f"DEBUG: Verification - stored result type: {type(stored)}")
                                    print(f"DEBUG: Verification - stored result success: {getattr(stored, 'success', 'N/A')}")
                                
                                # Force rerun to display results
                                st.rerun()
                            except Exception as store_error:
                                st.error(f"Failed to store result: {str(store_error)}")
                                import traceback
                                traceback.print_exc()
                        else:
                            st.error(f"Analysis failed: {result.error if result.error else 'Unknown error'}")
                            if result.error:
                                print(f"ERROR: {result.error}")
                    else:
                        st.error("Analysis returned no result")
                        print("ERROR: analyze_and_display returned None")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    import traceback
                    print(f"EXCEPTION: {e}")
                    traceback.print_exc()
                    # Don't rerun on error - let user see the error message
        
        # Display stored results (after rerun from button click)
        if 'analysis_result' in st.session_state:
            stored_result = st.session_state.get('analysis_result')
            input_text = st.session_state.get('analysis_input', '')
            
            print(f"DEBUG: Checking for stored result in session state")
            print(f"DEBUG: analysis_result key exists: {'analysis_result' in st.session_state}")
            print(f"DEBUG: Stored result is None: {stored_result is None}")
            print(f"DEBUG: Stored result type: {type(stored_result) if stored_result else 'N/A'}")
            
            if stored_result:
                result = stored_result
                try:
                    # Safely access similar_patents
                    if hasattr(result, 'similar_patents'):
                        similar_patents = result.similar_patents if result.similar_patents else []
                    else:
                        similar_patents = []
                        print(f"WARNING: Result missing similar_patents attribute")
                    
                    print(f"DEBUG: Retrieved result with {len(similar_patents)} similar patents")
                    print(f"DEBUG: Result success: {getattr(result, 'success', 'N/A')}")
                    print(f"DEBUG: Result novelty_score: {getattr(result, 'novelty_score', 'N/A')}")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("### Novelty Assessment")
                        render_novelty_result(result, input_text)
                    
                    with col2:
                        st.markdown("### Similar Patents Found")
                        render_prior_art(similar_patents, context="novelty", query_text=input_text)
                except Exception as e:
                    st.error(f"Error displaying results: {str(e)}")
                    import traceback
                    print(f"EXCEPTION in display: {e}")
                    traceback.print_exc()
                    # Clear the broken result
                    if 'analysis_result' in st.session_state:
                        del st.session_state['analysis_result']
            else:
                print(f"DEBUG: Result is None or falsy")
                if 'analysis_result' in st.session_state:
                    del st.session_state['analysis_result']
    
    with tab2:
        st.markdown("### Search Prior Art")
        
        search_query = st.text_input(
            "Search query:",
            placeholder="Enter keywords or a description to search for similar patents..."
        )
        
        if st.button("Search", type="primary"):
            if not search_query.strip():
                st.error("Please enter a search query.")
            else:
                log_messages = []
                log_expander = st.expander("Search Log", expanded=False)
                
                with log_expander:
                    log_placeholder = st.empty()
                
                def analyze_and_display(input_data, is_search=False):
                    def update_status(message: str):
                        log_messages.append(message)
                        log_placeholder.markdown(
                            '<div style="max-height: 300px; overflow-y: auto; font-family: monospace; font-size: 0.85rem; background: #252525; padding: 1rem; border-radius: 4px;">' +
                            '<br>'.join([f'<div style="margin: 0.25rem 0; color: #e0e0e0;">{msg}</div>' for msg in log_messages]) +
                            '</div>',
                            unsafe_allow_html=True
                        )
                    
                    serpapi_key = st.session_state.get('serpapi_key', '')
                    use_online = st.session_state.get('use_online', True)
                    use_keywords = st.session_state.get('use_keywords', True)
                    
                    update_status("Loading analyzer...")
                    analyzer = load_analyzer(
                        serpapi_key=serpapi_key if serpapi_key else None,
                        use_online=use_online,
                        use_keywords=use_keywords
                    )
                    
                    update_status("Starting search...")
                    result = analyzer.analyze(input_data, status_callback=update_status)
                    
                    update_status("Search complete!")
                    
                    return result
                
                try:
                    result = analyze_and_display(search_query, is_search=True)
                    if result:
                        st.session_state['search_result'] = result
                        st.session_state['last_query'] = search_query  # Store query for overlapping quotes
                    else:
                        st.error("Search returned no result")
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
                    import traceback
                    print(f"EXCEPTION: {e}")
                    traceback.print_exc()
                st.rerun()
        
        if 'search_result' in st.session_state and st.session_state.get('search_result'):
            result = st.session_state['search_result']
            st.markdown("### Search Results")
            similar_patents = result.similar_patents if result.similar_patents else []
            # Get query text from session state if available
            query_text = st.session_state.get('last_query', '')
            render_prior_art(similar_patents, max_display=20, context="search", query_text=query_text)


if __name__ == "__main__":
    main()
