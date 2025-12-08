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


def render_prior_art(patents: list, max_display: int = 10, context: str = "default"):
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
        sorted_patents.sort(key=lambda x: x.get('similarity_score', x.get('similarity', 0)), 
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
        if abstract_text and abstract_text != 'No abstract available':
            if len(abstract_text) > 800:
                abstract_text = abstract_text[:800].rsplit('.', 1)[0] + '.'
            abstract_text = abstract_text.strip()
            if abstract_text and not abstract_text.endswith(('.', '!', '?')):
                abstract_text += '.'
        abstract = html.escape(abstract_text)
        
        year = patent.get('year', 'N/A')
        if year == 'N/A' or not year or year == 0:
            grant_date = patent.get('grant_date', '') or patent.get('publication_date', '')
            if grant_date:
                year = str(grant_date)[:4] if len(str(grant_date)) >= 4 else 'N/A'
            else:
                year = 'N/A'
        else:
            year = str(year) if isinstance(year, (int, float)) else year
        score = patent.get('similarity', 0)
        if score == 0:
            score = patent.get('similarity_score', 0)
        
        source_badge = ""
        if patent.get('source') == 'online' or 'google' in str(patent_id).lower() or any(x in str(patent_id) for x in ['CN', 'JP', 'EP', 'WO']):
            source_badge = " [Online]"
        else:
            source_badge = " [Local]"
        
        with st.expander(f"**{title}** (Patent {patent_id}, {year}) - Similarity: {score:.3f}{source_badge}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Patent ID:** {patent_id}")
                st.markdown(f"**Year:** {year}")
                st.markdown(f"**Similarity Score:** {score:.3f}")
            with col2:
                if patent.get('cpc_code'):
                    st.markdown(f"**CPC Code:** {patent.get('cpc_code')}")
                if patent.get('assignee'):
                    st.markdown(f"**Assignee:** {patent.get('assignee')}")
                if patent.get('inventor'):
                    st.markdown(f"**Inventor:** {patent.get('inventor')}")
            
            st.markdown(f"**Abstract:** {abstract}")
            
            if patent.get('url') or patent.get('link'):
                patent_url = patent.get('url') or patent.get('link')
                st.markdown(f"**Link:** [{patent_url}]({patent_url})")
            
            if patent.get('claims'):
                st.markdown("**Claims:**")
                for claim in patent['claims'][:3]:
                    claim_str = str(claim) if not isinstance(claim, dict) else str(claim.get('text', claim))
                    st.markdown(f"- {html.escape(claim_str)}")


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
    lines.append(f"Assessment: {'Likely Novel' if score < 0.5 else 'Potential Prior Art Found'}")
    
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
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Novelty Score", f"{score:.3f}", 
                 delta="Novel" if score > 0.5 else "Not Novel", 
                 delta_color="normal" if score > 0.5 else "inverse")
    with col2:
        st.metric("Similar Patents", len(similar_patents))
    with col3:
        online_count = sum(1 for p in similar_patents if p.get('source') == 'online' or 'google' in str(p.get('patent_id', '')).lower())
        local_count = len(similar_patents) - online_count
        st.metric("Search Sources", f"{local_count} local, {online_count} online")
    
    score_class = "score-low" if score < 0.3 else "score-medium" if score < 0.7 else "score-high"
    
    if score > 0.7:
        interpretation = "Highly Novel"
        similarity_pct = f"{(1-score)*100:.0f}%"
    elif score > 0.5:
        interpretation = "Moderately Novel"
        similarity_pct = f"{(1-score)*100:.0f}%"
    elif score > 0.3:
        interpretation = "Low Novelty"
        similarity_pct = f"{(1-score)*100:.0f}%"
    else:
        interpretation = "Not Novel"
        similarity_pct = f"{(1-score)*100:.0f}%"
    
    st.markdown(f"""
    <div class="score-box">
        <div class="score-value {score_class}">{score:.2f}</div>
        <div class="score-label">Novelty Score</div>
        <div class="score-interpretation" style="margin-top: 0.5rem; font-size: 1rem; color: #aaa;">{interpretation}</div>
        <div class="score-similarity" style="margin-top: 0.25rem; font-size: 0.875rem; color: #888;">Similarity to Prior Art: {similarity_pct}</div>
    </div>
    <div style="margin-top: 0.5rem; padding: 0.75rem; background: #252525; border-radius: 4px; font-size: 0.85rem; color: #aaa;">
        <strong>Scale:</strong> 0.0 = Not Novel (100% similar to prior art) | 1.0 = Highly Novel (0% similar to prior art)<br>
        <strong>Your Score:</strong> {score:.2f} means your patent is {similarity_pct} similar to prior art, indicating {interpretation.lower()}.
    </div>
    """, unsafe_allow_html=True)
    
    if recommendation:
        if score < 0.3:
            st.success(f"**Recommendation:** {recommendation}")
        elif score < 0.7:
            st.warning(f"**Recommendation:** {recommendation}")
        else:
            st.error(f"**Recommendation:** {recommendation}")
    
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
        report_json = json.dumps({
            "novelty_score": score,
            "assessment": "Likely Novel" if score < 0.5 else "Potential Prior Art Found",
            "recommendation": recommendation,
            "explanation": explanation,
            "similar_patents": similar_patents,
            "search_metadata": result.search_metadata or {}
        }, indent=2)
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
                    
                result = analyze_and_display(idea_text)
                st.session_state['analysis_result'] = result
                st.session_state['analysis_input'] = idea_text
                st.rerun()
        
        if 'analysis_result' in st.session_state and st.session_state.get('analysis_result'):
            result = st.session_state['analysis_result']
            input_text = st.session_state.get('analysis_input', '')
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Novelty Assessment")
                render_novelty_result(result, input_text)
            
            with col2:
                st.markdown("### Similar Patents Found")
                similar_patents = result.similar_patents if result.similar_patents else []
                render_prior_art(similar_patents, context="novelty")
    
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
                
                result = analyze_and_display(search_query, is_search=True)
                st.session_state['search_result'] = result
                st.rerun()
        
        if 'search_result' in st.session_state and st.session_state.get('search_result'):
            result = st.session_state['search_result']
            st.markdown("### Search Results")
            similar_patents = result.similar_patents if result.similar_patents else []
            render_prior_art(similar_patents, max_display=20, context="search")


if __name__ == "__main__":
    main()
