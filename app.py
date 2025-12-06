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

# Add project root to path (deferred to avoid blocking)
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Page config - must be first Streamlit command, but optimized
st.set_page_config(
    page_title="Patent Novelty Analyzer",
    page_icon="",  # Empty icon for faster loading
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None  # Disable menu for faster startup
)

# CSS will be loaded in main() to avoid processing on every import

@st.cache_resource
def load_analyzer(serpapi_key: str = None, use_online: bool = True, use_keywords: bool = True):
    """Load the patent analyzer with Hybrid RAG settings."""
    from src.app.patent_analyzer import PatentAnalyzer
    
    return PatentAnalyzer(
        use_patentsview=True,
        use_full_phi3=True,
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


def render_prior_art(patents: list, max_display: int = 10):
    """Render prior art patents in cards."""
    import html
    
    if not patents:
        st.info("No similar patents found.")
        return
    
    for i, patent in enumerate(patents[:max_display]):
        patent_id = patent.get('patent_id', 'N/A')
        title = html.escape(patent.get('title', 'No title available'))
        abstract = html.escape(patent.get('abstract', 'No abstract available'))
        # Try multiple field names for year
        year = patent.get('year', 'N/A')
        if year == 'N/A' or not year or year == 0:
            # Try grant_date or publication_date
            grant_date = patent.get('grant_date', '') or patent.get('publication_date', '')
            if grant_date:
                year = str(grant_date)[:4] if len(str(grant_date)) >= 4 else 'N/A'
            else:
                year = 'N/A'
        else:
            # Convert to string if it's an integer
            year = str(year) if isinstance(year, (int, float)) else year
        # Try multiple field names for similarity score
        score = patent.get('similarity', 0)
        if score == 0:
            score = patent.get('similarity_score', 0)
        
        with st.expander(f"**{title}** (Patent {patent_id}, {year}) - Similarity: {score:.3f}", expanded=False):
            st.markdown(f"**Patent ID:** {patent_id}")
            st.markdown(f"**Year:** {year}")
            st.markdown(f"**Similarity Score:** {score:.3f}")
            st.markdown(f"**Abstract:** {abstract}")
            
            # Show full claims if available
            if patent.get('claims'):
                st.markdown("**Claims:**")
                for claim in patent['claims'][:3]:  # Show first 3 claims
                    st.markdown(f"- {html.escape(claim)}")


def render_novelty_result(result):
    """Render novelty assessment result."""
    score = result.novelty_score if result.novelty_score is not None else 0
    explanation = result.explanation if result.explanation else 'No explanation available.'
    similar_patents = result.similar_patents if result.similar_patents else []
    recommendation = result.recommendation if result.recommendation else ''
    
    # Score display
    score_class = "score-high" if score < 0.3 else "score-medium" if score < 0.7 else "score-low"
    st.markdown(f"""
    <div class="score-box">
        <div class="score-value {score_class}">{score:.2f}</div>
        <div class="score-label">Novelty Score (lower = more novel)</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendation
    if recommendation:
        if score < 0.3:
            st.success(f"**Recommendation:** {recommendation}")
        elif score < 0.7:
            st.warning(f"**Recommendation:** {recommendation}")
        else:
            st.error(f"**Recommendation:** {recommendation}")
    
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
    
    # Pre-load analyzer in background (optional - speeds up first query)
    # This happens automatically on first query, but we can pre-load here
    if 'analyzer_preloaded' not in st.session_state:
        # Don't block - let it load on first query
        # But mark that we're ready to load
        st.session_state['analyzer_preloaded'] = False
    
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Configuration")
        
        # SerpAPI key input
        serpapi_key = st.text_input(
            "SerpAPI Key (for online search)",
            value=st.session_state.get('serpapi_key', ''),
            type="password",
            help="Enter your SerpAPI key to enable online patent search via Google Patents"
        )
        
        # Update session state and environment
        if serpapi_key != st.session_state.get('serpapi_key', ''):
            st.session_state['serpapi_key'] = serpapi_key
            if serpapi_key:
                import os
                os.environ['SERPAPI_KEY'] = serpapi_key
                st.success("SerpAPI key configured")
                # Clear analyzer cache to reload with new key
                load_analyzer.clear()
            else:
                st.info("SerpAPI key removed")
                load_analyzer.clear()
        
        # Online search toggle
        use_online = st.checkbox(
            "Enable Online Search",
            value=st.session_state.get('use_online', True),
            help="Search Google Patents via SerpAPI for additional results"
        )
        st.session_state['use_online'] = use_online
        
        # LLM keyword generation toggle
        use_keywords = st.checkbox(
            "Use LLM Keyword Generation",
            value=st.session_state.get('use_keywords', True),
            help="Use Phi-3 to generate search keywords from input text"
        )
        st.session_state['use_keywords'] = use_keywords
        
        # Clear cache button
        if st.button("Clear Cache"):
            load_analyzer.clear()
            st.session_state['analyzer_preloaded'] = False
            st.success("Cache cleared!")
    
    # Main content
    tab1, tab2 = st.tabs(["Novelty Assessment", "Prior Art Search"])
    
    with tab1:
        st.markdown("### Enter Patent Information")
        
        idea_text = st.text_area(
            "Describe your invention or paste patent text:",
            height=200,
            placeholder="Enter a detailed description of your invention, including technical details, claims, and abstract..."
        )
        
        if st.button("Analyze Novelty", type="primary"):
            if not idea_text.strip():
                st.error("Please enter some text to analyze.")
            else:
                def analyze_and_display(input_data, is_search=False):
                    with st.status("🔍 Starting analysis...", expanded=True) as status:
                        log_container = st.container()
                        log_messages = []
                        
                        def update_status(message: str):
                            log_messages.append(message)
                            with log_container:
                                for msg in log_messages[-10:]:  # Show last 10 messages
                                    st.text(msg)
                            status.update(label=message, state="running")
                        
                        serpapi_key = st.session_state.get('serpapi_key', '')
                        use_online = st.session_state.get('use_online', True)
                        use_keywords = st.session_state.get('use_keywords', True)
                        
                        update_status("📦 Loading analyzer...")
                        analyzer = load_analyzer(
                            serpapi_key=serpapi_key if serpapi_key else None,
                            use_online=use_online,
                            use_keywords=use_keywords
                        )
                        
                        update_status("🚀 Starting analysis...")
                        result = analyzer.analyze(input_data, status_callback=update_status)
                        
                        status.update(label="✅ Analysis complete!", state="complete")
                    
                    # Display results
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("### Novelty Assessment")
                        render_novelty_result(result)
                    
                    with col2:
                        st.markdown("### Similar Patents Found")
                        similar_patents = result.similar_patents if result.similar_patents else []
                        render_prior_art(similar_patents)
                
                analyze_and_display(idea_text)
    
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
                def analyze_and_display(input_data, is_search=False):
                    with st.status("🔍 Starting search...", expanded=True) as status:
                        log_container = st.container()
                        log_messages = []
                        
                        def update_status(message: str):
                            log_messages.append(message)
                            with log_container:
                                for msg in log_messages[-10:]:  # Show last 10 messages
                                    st.text(msg)
                            status.update(label=message, state="running")
                        
                        serpapi_key = st.session_state.get('serpapi_key', '')
                        use_online = st.session_state.get('use_online', True)
                        use_keywords = st.session_state.get('use_keywords', True)
                        
                        update_status("📦 Loading analyzer...")
                        analyzer = load_analyzer(
                            serpapi_key=serpapi_key if serpapi_key else None,
                            use_online=use_online,
                            use_keywords=use_keywords
                        )
                        
                        update_status("🚀 Starting search...")
                        result = analyzer.analyze(input_data, status_callback=update_status)
                        
                        status.update(label="✅ Search complete!", state="complete")
                    
                    # Display results
                    st.markdown("### Search Results")
                    similar_patents = result.similar_patents if result.similar_patents else []
                    render_prior_art(similar_patents, max_display=20)
                
                analyze_and_display(search_query, is_search=True)


if __name__ == "__main__":
    main()
