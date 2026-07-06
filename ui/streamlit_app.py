import base64
import json
import re
import sys
import time
import uuid
from pathlib import Path

import requests
import streamlit as st

# ── eval module path setup (GPT-4o-mini-judged RAGAS scoring) ───────────────────
_ROOT     = Path(__file__).parents[1]
_EVAL_DIR = _ROOT / "eval"
for _p in (str(_EVAL_DIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ragas_judge import score_turn  # noqa: E402

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocIQ — Contract Intelligence",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API           = "http://localhost:8000"
EMBEDDING_DIM = 1536
PAGE_RE       = re.compile(r"\(page\s+(\d+)\)", re.IGNORECASE)

PIPELINE_STEPS = [
    "Parsing document layout",
    "Cleaning &amp; normalizing text",
    "Chunking into passages",
    "Generating embeddings",
    "Indexing in vector store",
    "Building keyword index",
]

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "screen":          "upload",
    "documents":       [],
    "upload_results":  [],
    "uploader_key":    0,
    "preview_bytes":   None,
    "preview_name":    None,
    "doc_previews":    {},
    "viewing_doc_id":  None,
    # Multi-session chat model (replaces single messages + chat_session_id)
    "chat_sessions":   {},    # {local_key: {title, messages, api_session_id, created_at}}
    "active_session":  None,  # local_key of currently displayed chat
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _new_chat_session() -> str:
    key = str(uuid.uuid4())
    st.session_state.chat_sessions[key] = {
        "title":          "New chat",
        "messages":       [],
        "api_session_id": None,
        "created_at":     time.time(),
    }
    st.session_state.active_session = key
    st.session_state.screen = "chat"
    return key


# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* ── global reset ── */
html, body, [class*="css"] {
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important;
  background-color:#111111;
}
html, body { background:#111111 !important; color:#f0f0f0 !important; }
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.appview-container, section.main,
.main .block-container { background:#111111 !important; }
[data-testid="stHeader"] { display:none !important; }
#MainMenu, footer { display:none !important; }
.block-container { padding-top:0 !important; padding-left:1.2rem !important;
                   padding-right:1.2rem !important; max-width:100% !important; }

/* ── sidebar shell ── */
[data-testid="stSidebar"] {
  background:#161616 !important;
  border-right:0.5px solid #252525 !important;
  min-width:280px !important; max-width:280px !important;
}
[data-testid="stSidebar"] > div:first-child {
  width:280px !important; min-width:280px !important;
}
[data-testid="stSidebarContent"] { padding:0 !important; }

/* ── sidebar default buttons (nav items) ── */
[data-testid="stSidebar"] .stButton > button {
  background:transparent !important;
  color:#d8d8d8 !important;
  border:none !important;
  border-left:2px solid transparent !important;
  border-radius:0 !important;
  width:100% !important;
  text-align:left !important;
  padding:8px 16px !important;
  font-size:15px !important;
  font-weight:400 !important;
  font-family:'Inter',sans-serif !important;
  box-shadow:none !important;
  transition:all .12s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background:#1f1f1f !important;
  color:#ffffff !important;
}

/* ── New Chat button (prominent, top of sidebar) ── */
[data-testid="stSidebar"] .sidebar-new-chat .stButton > button {
  background:rgba(245,197,24,.1) !important;
  color:#F5C518 !important;
  border:0.5px solid rgba(245,197,24,.28) !important;
  border-left:0.5px solid rgba(245,197,24,.28) !important;
  border-radius:8px !important;
  margin:6px 12px 2px !important;
  width:calc(100% - 24px) !important;
  padding:8px 14px !important;
  font-size:15px !important;
  font-weight:500 !important;
}
[data-testid="stSidebar"] .sidebar-new-chat .stButton > button:hover {
  background:rgba(245,197,24,.17) !important;
  border-color:rgba(245,197,24,.5) !important;
  color:#ffd835 !important;
}

/* ── Doc intelligence nav button ── */
[data-testid="stSidebar"] .nav-doc .stButton > button {
  font-size:15px !important;
  padding:7px 14px !important;
  color:#d8d8d8 !important;
}
[data-testid="stSidebar"] .nav-doc .stButton > button:hover {
  background:#1f1f1f !important;
  color:#ffffff !important;
}

/* ── chat list item — default ── */
[data-testid="stSidebar"] .chat-item-btn .stButton > button {
  padding:7px 12px !important;
  font-size:15px !important;
  color:#c8c8c8 !important;
  border-left:2px solid transparent !important;
  border-radius:0 !important;
  white-space:nowrap !important;
  overflow:hidden !important;
  text-overflow:ellipsis !important;
}
[data-testid="stSidebar"] .chat-item-btn .stButton > button:hover {
  background:#1e1e1e !important;
  color:#ffffff !important;
}

/* ── chat list item — active ── */
[data-testid="stSidebar"] .chat-item-active .stButton > button {
  background:rgba(245,197,24,.07) !important;
  color:#ffffff !important;
  border-left:2px solid #F5C518 !important;
  border-radius:0 !important;
  padding:7px 12px !important;
  font-size:15px !important;
  font-weight:500 !important;
}

/* ── delete (×) button ── */
[data-testid="stSidebar"] .chat-del-btn .stButton > button {
  background:transparent !important;
  color:#333 !important;
  border:none !important;
  border-left:none !important;
  border-radius:4px !important;
  padding:0 !important;
  font-size:14px !important;
  min-height:26px !important;
  height:26px !important;
  width:24px !important;
  line-height:1 !important;
  display:flex !important;
  align-items:center !important;
  justify-content:center !important;
}
[data-testid="stSidebar"] .chat-del-btn .stButton > button:hover {
  color:#ff5252 !important;
  background:rgba(255,80,80,.1) !important;
}

/* ── sidebar user card — inset card, normal flow (no absolute) ── */
#sidebar-user-card {
  border:0.5px solid #252525; border-radius:10px;
  padding:11px 13px; margin:12px 10px 10px;
  display:flex; align-items:center; gap:10px;
  background:#111111; box-sizing:border-box;
  width:calc(100% - 20px);
}

/* ── main yellow buttons ── */
[data-testid="stMainBlockContainer"] .stButton > button {
  background:#F5C518 !important; color:#1a1500 !important;
  border:none !important; border-radius:8px !important;
  font-weight:500 !important; font-size:13px !important;
}
[data-testid="stMainBlockContainer"] .stButton > button:hover { background:#D4A900 !important; }
[data-testid="stMainBlockContainer"] .stButton > button:disabled {
  background:#3a3a2a !important; color:#666 !important;
  cursor:not-allowed !important; opacity:1 !important;
}

/* ── back button ── */
.back-btn .stButton > button {
  background:transparent !important; color:#aaa !important;
  border:0.5px solid #3a3a3a !important; border-radius:8px !important;
  font-size:16px !important; padding:4px 10px !important;
  min-height:32px !important; height:32px !important;
  line-height:1 !important; font-weight:300 !important;
}
.back-btn .stButton > button:hover {
  color:#f0f0f0 !important; border-color:#f0f0f0 !important;
  background:rgba(255,255,255,.04) !important;
}

/* ── new chat button in chat header ── */
.new-chat-btn .stButton > button {
  background:transparent !important;
  color:#F5C518 !important;
  border:0.5px solid rgba(245,197,24,.4) !important;
  border-radius:8px !important;
  font-size:12px !important; font-weight:500 !important;
  padding:4px 12px !important;
  min-height:32px !important; height:32px !important;
  width:100% !important; white-space:nowrap !important;
}
.new-chat-btn .stButton > button:hover {
  background:rgba(245,197,24,.08) !important;
  border-color:#F5C518 !important;
  color:#ffd025 !important;
}

/* ── ready-chat CTA ── */
.ready-chat-btn .stButton > button {
  background:linear-gradient(135deg,#F5C518,#D4A900) !important;
  color:#1a1500 !important; border:none !important; border-radius:10px !important;
  width:100% !important; min-height:58px !important;
  font-size:15px !important; font-weight:600 !important;
  box-shadow:0 4px 20px rgba(245,197,24,.2) !important;
}
.ready-chat-btn .stButton > button:hover {
  background:linear-gradient(135deg,#ffd025,#c49a00) !important;
}

/* ── file uploader ── */
[data-testid="stFileUploaderDropzone"] {
  background:#161616 !important;
  border:1.5px dashed rgba(245,197,24,.5) !important;
  border-radius:12px !important; padding:24px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] span { color:#666 !important; }
[data-testid="stFileUploader"] > label { color:#f0f0f0 !important; font-size:13px !important; }
[data-testid="stFileUploaderDropzone"] button {
  background:#F5C518 !important; color:#1a1500 !important;
  border:none !important; border-radius:6px !important; font-weight:500 !important;
}

/* ── View button (vb-marker trick) ── */
div:has(.vb-marker) ~ div button,
div:has(.vb-marker) + div button {
  background:transparent !important;
  border:0.5px solid rgba(245,197,24,.35) !important;
  color:#F5C518 !important; border-radius:6px !important;
  font-size:11px !important; padding:4px 10px !important;
  font-weight:400 !important; min-height:0 !important; height:auto !important;
}
div:has(.vb-marker) ~ div button:hover,
div:has(.vb-marker) + div button:hover {
  background:rgba(245,197,24,.08) !important; border-color:#F5C518 !important;
}

/* ── chat messages ── */
[data-testid="stChatMessage"] {
  background:#171717 !important; border:0.5px solid #222 !important;
  border-radius:12px !important; margin-bottom:8px !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  border-left:2px solid rgba(245,197,24,.4) !important;
  background:rgba(245,197,24,.02) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
  border-left:2px solid rgba(100,160,255,.22) !important;
}

/* ── chat input bar ── */
[data-testid="stBottom"],
section[data-testid="stBottom"],
.stBottom,
[class*="stBottom"] {
  background:#111111 !important;
  background-color:#111111 !important;
  border-top:none !important;
  box-shadow:none !important;
  padding-top:8px !important;
  padding-bottom:12px !important;
  padding-left:0 !important;
  padding-right:0 !important;
  width:100% !important;
  max-width:100% !important;
}
[data-testid="stChatInput"],
[data-testid="stChatInput"] *:not(textarea):not(button):not(svg):not(path) {
  background:#111111 !important;
  background-color:#111111 !important;
  border:none !important;
  box-shadow:none !important;
}
[data-testid="stChatInput"] {
  width:100% !important;
  max-width:100% !important;
  padding:0 1.2rem !important;
}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] form {
  background:#1c1c1c !important;
  border:0.5px solid #2a2a2a !important;
  border-radius:12px !important;
  display:flex !important;
  align-items:center !important;
  padding:6px 6px 6px 14px !important;
  gap:6px !important;
  width:100% !important;
  box-shadow:none !important;
}
[data-testid="stChatInput"] > div:focus-within,
[data-testid="stChatInput"] form:focus-within {
  border-color:rgba(245,197,24,.4) !important;
  box-shadow:0 0 0 2px rgba(245,197,24,.05) !important;
}
[data-testid="stChatInput"] textarea {
  background:transparent !important;
  background-color:transparent !important;
  color:#f0f0f0 !important;
  border:none !important;
  box-shadow:none !important;
  border-radius:0 !important;
  caret-color:#F5C518 !important;
  flex:1 !important;
  resize:none !important;
  padding:6px 0 !important;
  min-height:36px !important;
}
[data-testid="stChatInput"] textarea:focus {
  border:none !important;
  box-shadow:none !important;
  background:transparent !important;
  outline:none !important;
}
[data-testid="stChatInput"] textarea::placeholder { color:#444 !important; }
[data-testid="stChatInput"] button {
  background:#F5C518 !important;
  color:#1a1500 !important;
  border:none !important;
  border-radius:8px !important;
  flex-shrink:0 !important;
  margin-left:auto !important;
  height:36px !important;
  width:36px !important;
  padding:0 !important;
  display:flex !important;
  align-items:center !important;
  justify-content:center !important;
}
[data-testid="stChatInput"] button:hover { background:#D4A900 !important; }

/* ── source chips ── */
.src-chip {
  display:inline-block;
  background:rgba(245,197,24,.07); color:#c8a214;
  border:0.5px solid rgba(245,197,24,.2); border-radius:4px;
  padding:2px 8px; font-size:10px; margin:0 3px 2px 0;
  font-family:monospace; letter-spacing:0.2px;
}

/* ── misc ── */
[data-testid="stAlert"] {
  background:rgba(245,197,24,.08) !important;
  border:0.5px solid rgba(245,197,24,.25) !important; border-radius:10px !important;
}
[data-testid="stSelectbox"] > div > div {
  background:#1a1a1a !important; border:0.5px solid #3a3a3a !important;
  color:#f0f0f0 !important; border-radius:8px !important;
}
h1,h2,h3,h4,h5,h6 { color:#f0f0f0 !important; }
[data-testid="stMarkdown"] p { color:#f0f0f0 !important; }
.stCaption p { color:#999 !important; font-size:11px !important; }
hr { border-color:#2e2e2e !important; margin:12px 0 !important; }

/* ── sidebar column gap fix ── */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
  gap:0 !important;
}

/* ── evaluate button (RAGAS · GPT-4o mini) ── */
.eval-btn .stButton > button {
  background:transparent !important;
  border:0.5px solid rgba(38,198,218,.35) !important;
  color:#26C6DA !important; border-radius:6px !important;
  font-size:11px !important; padding:4px 10px !important;
  font-weight:400 !important; min-height:0 !important; height:auto !important;
  width:auto !important; margin-top:6px !important;
}
.eval-btn .stButton > button:hover {
  background:rgba(38,198,218,.08) !important; border-color:#26C6DA !important;
}

/* ── RAGAS metric cards ── */
.ragas-card {
  background:#171717; border:0.5px solid #262626; border-radius:10px;
  padding:10px 14px; text-align:center; flex:1; min-width:0;
}
.ragas-chunk {
  background:#161616; border:0.5px solid #232323; border-radius:8px;
  padding:8px 12px; margin-bottom:6px; font-size:12px; color:#ccc;
  line-height:1.5; word-break:break-word;
}

/* ── animations ── */
@keyframes pipePulse {
  0%, 100% { opacity:1; transform:scale(1.15); }
  50%       { opacity:0.35; transform:scale(0.8); }
}
@keyframes bounce {
  0%,80%,100% { transform:translateY(0); }
  40%          { transform:translateY(-6px); }
}
@keyframes glow {
  0%, 100% { transform:scale(1);    opacity:0.65; }
  50%       { transform:scale(1.35); opacity:0.25; }
}
</style>
""", unsafe_allow_html=True)

# ── Active nav highlight (Doc intelligence only) ───────────────────────────────
if st.session_state.screen == "upload":
    st.markdown(
        "<style>[data-testid='stSidebar'] .nav-doc .stButton > button {"
        " background:rgba(245,197,24,.08) !important; color:#F5C518 !important;"
        " border-left:2px solid #F5C518 !important; font-weight:500 !important;}</style>",
        unsafe_allow_html=True,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _pipeline_card(
    name, size_bytes, page_count, chunk_count, step_statuses,
    show_ready=False, show_steps=True,
):
    size_kb  = round(size_bytes / 1024, 1)
    page_str = f"{page_count} page{'s' if page_count != 1 else ''}" if page_count else "—"

    ready_badge = ""
    if show_ready:
        ready_badge = (
            '<span style="display:inline-flex;align-items:center;gap:5px;'
            'background:rgba(90,200,120,.1);color:#5ac878;'
            'border:0.5px solid rgba(90,200,120,.3);border-radius:20px;'
            'padding:3px 11px;font-size:10px;font-weight:500;flex-shrink:0;">'
            '<span style="width:5px;height:5px;border-radius:50%;'
            'background:#5ac878;display:inline-block;"></span>Ready</span>'
        )

    steps_html = dots_html = ""
    if show_steps:
        rows = ""
        for step, status in zip(PIPELINE_STEPS, step_statuses):
            if status == "done":
                dot   = '<div style="width:10px;height:10px;border-radius:50%;background:#5ac878;flex-shrink:0;"></div>'
                color = "#bbb"
            elif status == "running":
                dot   = '<div style="width:10px;height:10px;border-radius:50%;background:#F5C518;flex-shrink:0;animation:pipePulse 0.85s ease-in-out infinite;"></div>'
                color = "#F5C518"
            else:
                dot   = '<div style="width:10px;height:10px;border-radius:50%;background:#222;border:1px solid #333;flex-shrink:0;"></div>'
                color = "#3a3a3a"
            rows += (
                f'<div style="display:flex;align-items:center;gap:8px;font-size:12px;'
                f'color:{color};padding:3px 0;">{dot}{step}</div>'
            )
        done_n  = sum(1 for s in step_statuses if s == "done")
        filled  = done_n * 16 // 6
        dot_row = "".join(
            f'<span style="width:8px;height:8px;border-radius:50%;display:inline-block;'
            f'margin-right:3px;background:{"#F5C518" if i < filled else "#222"};'
            f'{"border:1px solid #2e2e2e;" if i >= filled else ""}"></span>'
            for i in range(16)
        )
        steps_html = f'<div style="border-top:1.5px solid #F5C518;margin-bottom:10px;"></div>{rows}'
        dots_html  = f'<div style="margin-top:10px;line-height:1;">{dot_row}</div>'

    stats = ""
    if chunk_count:
        stats = (
            f'<div style="margin-top:10px;font-size:11px;display:flex;gap:20px;flex-wrap:wrap;'
            f'padding-top:8px;border-top:0.5px solid #1e1e1e;">'
            f'<span><b style="color:#F5C518;">{chunk_count}</b> <span style="color:#666;">chunks</span></span>'
            f'<span><b style="color:#F5C518;">{chunk_count}</b> <span style="color:#666;">embeddings</span></span>'
            f'<span><b style="color:#F5C518;">{EMBEDDING_DIM}</b><span style="color:#666;">-dim</span></span>'
            f'<span><b style="color:#F5C518;">1</b> <span style="color:#666;">vector shard</span></span></div>'
        )

    mb = "12px" if show_steps else "0"
    return f"""
<div style="background:#1a1a1a;border:0.5px solid #252525;border-radius:10px;
            padding:14px 16px;margin-bottom:4px;">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:{mb};">
    <div style="width:40px;height:40px;border-radius:8px;background:#222;
                display:flex;align-items:center;justify-content:center;
                flex-shrink:0;font-size:18px;border:0.5px solid #2e2e2e;">📄</div>
    <div style="flex:1;min-width:0;">
      <div style="font-size:13px;font-weight:500;color:#f0f0f0;
                  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</div>
      <div style="font-size:11px;color:#555;margin-top:2px;">{page_str} · {size_kb} KB</div>
    </div>
    {ready_badge}
  </div>
  {steps_html}{dots_html}{stats}
</div>"""


def _pdf_preview(pdf_bytes, filename):
    b64     = base64.b64encode(pdf_bytes).decode()
    size_kb = round(len(pdf_bytes) / 1024, 1)
    return f"""
<div style="background:#1a1a1a;border:0.5px solid #252525;border-radius:10px;
            overflow:hidden;display:flex;flex-direction:column;height:580px;">
  <div style="padding:10px 14px;border-bottom:0.5px solid #252525;
              display:flex;align-items:center;gap:8px;flex-shrink:0;background:#161616;">
    <span style="font-size:15px;">📄</span>
    <div style="flex:1;min-width:0;">
      <div style="font-size:12px;font-weight:500;color:#f0f0f0;
                  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{filename}</div>
      <div style="font-size:10px;color:#555;">{size_kb} KB · PDF preview</div>
    </div>
    <span style="font-size:10px;color:#555;">📌 pinned</span>
  </div>
  <iframe src="data:application/pdf;base64,{b64}#toolbar=0&navpanes=0"
          style="flex:1;border:none;background:#fff;" width="100%"></iframe>
</div>"""


def _thinking_html():
    steps = [
        ("🔍", "Searching document corpus"),
        ("📊", "Ranking semantic matches"),
        ("🧩", "Extracting relevant context"),
        ("✍️", "Composing response"),
    ]
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:7px;font-size:11px;color:#555;padding:2px 0;">'
        f'<span style="opacity:0.7;">{icon}</span><span>{text}</span></div>'
        for icon, text in steps
    )
    dots = "".join(
        f'<span style="width:6px;height:6px;border-radius:50%;background:#F5C518;'
        f'animation:bounce 1.4s {d} ease-in-out infinite;display:inline-block;margin-right:4px;"></span>'
        for d in ["0s", ".25s", ".5s"]
    )
    return (
        f'<div style="background:linear-gradient(135deg,#1a1a1a,#1c1a14);'
        f'border:0.5px solid rgba(245,197,24,.12);border-radius:4px 12px 12px 12px;'
        f'padding:14px 16px;max-width:88%;">'
        f'{rows}'
        f'<div style="margin-top:10px;display:flex;align-items:center;gap:5px;">'
        f'{dots}<span style="font-size:10px;color:#444;">Thinking…</span>'
        f'</div></div>'
    )


def _pages_from_source(source: str) -> list[int]:
    return [int(m) for m in PAGE_RE.findall(source or "")]


def _ragas_color(value):
    if value is None:
        return "#666666"
    return "#4CAF50" if value >= 0.8 else ("#FF9800" if value >= 0.5 else "#F44336")


def _latency_color(seconds):
    if seconds is None:
        return "#666666"
    return "#4CAF50" if seconds < 5 else ("#FF9800" if seconds < 15 else "#F44336")


def _metric_card_html(label: str, display: str, color: str) -> str:
    return (
        f'<div class="ragas-card"><div style="font-size:20px;font-weight:600;color:{color};">'
        f'{display}</div><div style="font-size:10px;color:#888;text-transform:uppercase;'
        f'letter-spacing:.05em;margin-top:3px;">{label}</div></div>'
    )


def _pipeline_metrics_html(msg: dict) -> str:
    confidence    = msg.get("intent_confidence", 0.0)
    latency_s     = msg.get("llm_latency_ms", 0.0) / 1000
    input_tokens  = msg.get("input_tokens", 0)
    output_tokens = msg.get("output_tokens", 0)

    cards = "".join([
        _metric_card_html("Intent Confidence", f"{confidence:.0%}", _ragas_color(confidence)),
        _metric_card_html("LLM Latency", f"{latency_s:.2f}s", _latency_color(latency_s)),
        _metric_card_html("Input Tokens", f"{input_tokens:,}", "#F5C518"),
        _metric_card_html("Output Tokens", f"{output_tokens:,}", "#F5C518"),
    ])
    return (
        '<div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;'
        'letter-spacing:.06em;margin:8px 0 6px;">Pipeline metrics</div>'
        f'<div style="display:flex;gap:10px;margin-bottom:4px;">{cards}</div>'
    )


def _ragas_metrics_html(score) -> str:
    metrics = [
        ("Faithfulness", score.faithfulness),
        ("Answer Relevancy", score.answer_relevancy),
        ("Context Precision", score.context_precision),
        ("Context Recall", score.context_recall),
    ]
    cards = "".join(
        f'<div class="ragas-card">'
        f'<div style="font-size:20px;font-weight:600;color:{_ragas_color(v)};">'
        f'{f"{v:.2f}" if v is not None else "—"}</div>'
        f'<div style="font-size:10px;color:#888;text-transform:uppercase;'
        f'letter-spacing:.05em;margin-top:3px;">{label}</div></div>'
        for label, v in metrics
    )
    return (
        f'<div style="font-size:10px;color:#26C6DA;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:.06em;margin:10px 0 6px;">RAGAS · judged by GPT-4o mini</div>'
        f'<div style="display:flex;gap:10px;margin-bottom:10px;">{cards}</div>'
    )


def _ragas_chunks_html(chunks: list) -> str:
    if not chunks:
        return '<div style="color:#666;font-size:12px;">No chunks were retrieved for this answer.</div>'
    rows = ""
    for i, c in enumerate(chunks, 1):
        source  = c.get("source", "document")
        content = (c.get("content") or "").strip()
        pages   = _pages_from_source(source)
        page_badges = "".join(
            f'<span style="background:rgba(245,197,24,.12);color:#F5C518;'
            f'border-radius:4px;padding:1px 7px;font-size:10px;margin-left:6px;">'
            f'page {p}</span>' for p in pages
        )
        preview = content[:500] + ("…" if len(content) > 500 else "")
        rows += (
            f'<div class="ragas-chunk"><div style="color:#888;font-size:10px;'
            f'margin-bottom:4px;">#{i} · {source}{page_badges}</div>{preview}</div>'
        )
    return (
        '<div style="font-size:10px;color:#888;font-weight:600;text-transform:uppercase;'
        'letter-spacing:.06em;margin:2px 0 6px;">Retrieved chunks &amp; PDF pages</div>'
        f'{rows}'
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Logo / brand ──────────────────────────────────────────────────────
    st.markdown("""
<div style="padding:16px 14px 14px;border-bottom:0.5px solid #222;
            display:flex;align-items:center;gap:10px;margin-bottom:2px;">
  <div style="width:30px;height:30px;background:linear-gradient(135deg,#F5C518,#c49a00);
              border-radius:8px;display:flex;align-items:center;justify-content:center;
              font-size:16px;flex-shrink:0;">💬</div>
  <div>
    <div style="font-size:15px;font-weight:600;color:#f0f0f0;letter-spacing:0.2px;">DocIQ</div>
    <div style="font-size:13px;color:#aaa;">Document intelligence</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── New Chat button ───────────────────────────────────────────────────
    st.markdown('<div class="sidebar-new-chat">', unsafe_allow_html=True)
    if st.button("＋  New Chat", key="sidebar_new_chat"):
        _new_chat_session()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Workspace section ─────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:12px;color:#bbb;font-weight:600;letter-spacing:0.9px;'
        'text-transform:uppercase;padding:10px 14px 2px;">Workspace</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nav-doc">', unsafe_allow_html=True)
    if st.button("  📂  Doc intelligence", key="nav_upload"):
        st.session_state.screen = "upload"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Chat sessions list ────────────────────────────────────────────────
    if st.session_state.chat_sessions:
        st.markdown(
            '<div style="font-size:12px;color:#bbb;font-weight:600;letter-spacing:0.9px;'
            'text-transform:uppercase;padding:10px 14px 2px;">Chats</div>',
            unsafe_allow_html=True,
        )
        sessions_sorted = sorted(
            st.session_state.chat_sessions.items(),
            key=lambda x: x[1]["created_at"],
            reverse=True,
        )
        for s_key, session in sessions_sorted:
            is_active = s_key == st.session_state.active_session
            c_title, c_del = st.columns([11, 1])
            with c_title:
                css_cls = "chat-item-active" if is_active else "chat-item-btn"
                st.markdown(f'<div class="{css_cls}">', unsafe_allow_html=True)
                raw = session["title"]
                label = raw[:27] + "…" if len(raw) > 27 else raw
                if st.button(label, key=f"sess_{s_key}"):
                    st.session_state.active_session = s_key
                    st.session_state.screen = "chat"
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with c_del:
                st.markdown('<div class="chat-del-btn">', unsafe_allow_html=True)
                if st.button("×", key=f"del_{s_key}"):
                    del st.session_state.chat_sessions[s_key]
                    if st.session_state.active_session == s_key:
                        remaining = sorted(
                            st.session_state.chat_sessions.keys(),
                            key=lambda k: st.session_state.chat_sessions[k]["created_at"],
                            reverse=True,
                        )
                        if remaining:
                            st.session_state.active_session = remaining[0]
                            st.session_state.screen = "chat"
                        else:
                            st.session_state.active_session = None
                            st.session_state.screen = "upload"
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    # ── Contract IQ card ─────────────────────────────────────────────────
    st.markdown("""
<div id="sidebar-user-card">
  <div style="width:28px;height:28px;border-radius:50%;
              background:linear-gradient(135deg,#F5C518,#c49a00);
              display:flex;align-items:center;justify-content:center;
              font-size:10px;font-weight:600;color:#1a1500;flex-shrink:0;">CI</div>
  <div>
    <div style="font-size:15px;color:#ffffff;font-weight:500;">Contract IQ</div>
    <div style="font-size:13px;color:#aaa;">Pro plan</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Screen 1 — Document intelligence / Upload
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "upload":

    st.markdown("""
<div style="padding:14px 0 12px;border-bottom:0.5px solid #1e1e1e;margin-bottom:14px;">
  <h2 style="font-size:15px;font-weight:500;margin:0;color:#f0f0f0;">Document intelligence</h2>
</div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([54, 46])

    with col_left:
        uploaded_files = st.file_uploader(
            "Drop your documents here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="visible",
            key=f"uploader_{st.session_state.uploader_key}",
            help="PDF — up to 50 MB per file",
        )

        if uploaded_files:
            st.session_state.preview_bytes = uploaded_files[0].getvalue()
            st.session_state.preview_name  = uploaded_files[0].name

        process_clicked = st.button(
            "Upload & Process",
            disabled=not bool(uploaded_files),
            key="process_btn",
        )

        # ── Live SSE streaming ─────────────────────────────────────────
        if process_clicked and uploaded_files:
            st.session_state.upload_results = []
            main_area = st.empty()

            for uf in uploaded_files:
                uf_bytes   = uf.getvalue()
                uf_name    = uf.name
                size_bytes = len(uf_bytes)

                st.session_state.preview_bytes = uf_bytes
                st.session_state.preview_name  = uf_name

                step_statuses = ["pending"] * 6
                page_count    = 0
                chunk_count   = 0

                main_area.markdown(
                    _pipeline_card(uf_name, size_bytes, page_count, chunk_count,
                                   step_statuses, show_steps=True),
                    unsafe_allow_html=True,
                )

                try:
                    with requests.post(
                        f"{API}/upload/stream",
                        files=[("files", (uf_name, uf_bytes, "application/pdf"))],
                        stream=True, timeout=180,
                    ) as resp:
                        resp.raise_for_status()
                        for raw_line in resp.iter_lines():
                            if not raw_line:
                                continue
                            line = raw_line.decode("utf-8")
                            if not line.startswith("data: "):
                                continue
                            event = json.loads(line[6:])

                            if "error" in event:
                                st.session_state.upload_results.append(("error", event["error"]))
                                break

                            if "step" in event:
                                idx = event["step"]
                                s   = event["status"]
                                if s == "running": step_statuses[idx] = "running"
                                elif s == "done":  step_statuses[idx] = "done"
                                if "pages"  in event: page_count  = event["pages"]
                                if "chunks" in event: chunk_count = event["chunks"]

                            if "done" in event:
                                result      = event["file"]
                                page_count  = result.get("page_count",  0)
                                chunk_count = result.get("chunk_count", 0)
                                file_id     = result["file_id"]
                                step_statuses = ["done"] * 6
                                st.session_state.documents.append(result)
                                st.session_state.upload_results.append(("success", result))
                                st.session_state.doc_previews[file_id] = uf_bytes

                            main_area.markdown(
                                _pipeline_card(uf_name, size_bytes, page_count, chunk_count,
                                               step_statuses, show_steps=True),
                                unsafe_allow_html=True,
                            )

                except requests.exceptions.ConnectionError:
                    st.session_state.upload_results.append((
                        "error", "Cannot reach http://localhost:8000 — run `python main.py` first.",
                    ))
                    break
                except Exception as exc:
                    st.session_state.upload_results.append(("error", f"✗ {uf_name}: {exc}"))

            st.session_state.uploader_key += 1
            st.rerun()

        else:
            for kind, payload in st.session_state.upload_results:
                if kind == "error":
                    st.error(payload)

            if st.session_state.documents:
                st.markdown(
                    '<div style="font-size:10px;color:#555;font-weight:600;letter-spacing:0.8px;'
                    'text-transform:uppercase;padding:4px 0 10px;">Uploaded Documents</div>',
                    unsafe_allow_html=True,
                )
                for doc in st.session_state.documents:
                    c_card, c_btn = st.columns([9, 1])
                    with c_card:
                        st.markdown(
                            _pipeline_card(
                                doc["original_name"], doc["size_bytes"],
                                doc.get("page_count", 0), doc.get("chunk_count", 0),
                                ["done"] * 6, show_ready=True, show_steps=False,
                            ),
                            unsafe_allow_html=True,
                        )
                    with c_btn:
                        st.markdown('<div class="vb-marker"></div>', unsafe_allow_html=True)
                        if st.button("View", key=f"view_{doc['file_id']}"):
                            fid = doc["file_id"]
                            if fid in st.session_state.doc_previews:
                                st.session_state.preview_bytes  = st.session_state.doc_previews[fid]
                                st.session_state.preview_name   = doc["original_name"]
                                st.session_state.viewing_doc_id = fid
                            st.rerun()

            elif not st.session_state.upload_results:
                st.markdown("""
<div style="display:flex;flex-direction:column;align-items:center;text-align:center;
            padding:34px 16px;">
  <div style="font-size:36px;margin-bottom:12px;opacity:0.2;">📂</div>
  <span style="font-size:13px;color:#555;font-weight:500;">No documents uploaded yet</span>
  <small style="font-size:11px;margin-top:5px;color:#444;">Select a PDF above to get started</small>
</div>""", unsafe_allow_html=True)

        if st.session_state.documents:
            st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
            if st.button("Clear all documents", key="clear_docs"):
                st.session_state.documents      = []
                st.session_state.upload_results = []
                st.session_state.preview_bytes  = None
                st.session_state.preview_name   = None
                st.session_state.doc_previews   = {}
                st.session_state.viewing_doc_id = None
                st.session_state.chat_sessions  = {}
                st.session_state.active_session = None
                st.rerun()

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        if st.session_state.documents:
            n = len(st.session_state.documents)
            st.markdown('<div class="ready-chat-btn">', unsafe_allow_html=True)
            if st.button(f"Ready to chat  ·  {n} document{'s' if n != 1 else ''} processed",
                         use_container_width=True, key="cta_chat"):
                _new_chat_session()
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.button("Upload a document to get started", disabled=True, use_container_width=True)

    with col_right:
        preview_bytes = None
        preview_name  = None
        vid = st.session_state.viewing_doc_id
        if vid and vid in st.session_state.doc_previews:
            preview_bytes = st.session_state.doc_previews[vid]
            for d in st.session_state.documents:
                if d["file_id"] == vid:
                    preview_name = d["original_name"]
                    break
        else:
            preview_bytes = st.session_state.preview_bytes
            preview_name  = st.session_state.preview_name

        if preview_bytes:
            st.markdown(_pdf_preview(preview_bytes, preview_name or ""), unsafe_allow_html=True)
        else:
            st.markdown("""
<div style="background:#161616;border:1px dashed #1e1e1e;border-radius:10px;
            height:480px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;gap:10px;">
  <div style="font-size:44px;opacity:0.1;">📄</div>
  <div style="font-size:12px;color:#2a2a2a;font-weight:500;">PDF preview</div>
  <div style="font-size:11px;color:#222;">Click View on a document, or select a file above</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Screen 2 — Chat
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "chat":

    # Ensure we always have an active session object to render
    active_key = st.session_state.active_session
    if not active_key or active_key not in st.session_state.chat_sessions:
        if st.session_state.documents:
            active_key = _new_chat_session()
        else:
            st.markdown("""
<div style="text-align:center;padding:60px 16px;">
  <div style="font-size:36px;margin-bottom:12px;">⚠️</div>
  <p style="font-size:15px;font-weight:500;color:#f0f0f0;">No documents indexed yet</p>
  <p style="font-size:12px;color:#555;margin-top:6px;">Go back and upload a PDF to get started.</p>
</div>""", unsafe_allow_html=True)
            st.stop()

    session  = st.session_state.chat_sessions[active_key]
    messages = session["messages"]

    # ── Header ─────────────────────────────────────────────────────────────
    c_back, c_select, c_new = st.columns([1, 5, 2])
    with c_back:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("←", key="nav_back", help="Back to documents"):
            st.session_state.screen = "upload"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with c_select:
        doc_names = ["All documents"] + [d["original_name"] for d in st.session_state.documents]
        st.selectbox("Document scope", doc_names, label_visibility="collapsed", key="doc_scope")
    with c_new:
        st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
        if st.button("＋ New Chat", key="new_chat_btn"):
            _new_chat_session()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if not st.session_state.documents:
        st.markdown("""
<div style="text-align:center;padding:60px 16px;">
  <div style="font-size:36px;margin-bottom:12px;">⚠️</div>
  <p style="font-size:15px;font-weight:500;color:#f0f0f0;">No documents indexed yet</p>
  <p style="font-size:12px;color:#555;margin-top:6px;">Go back and upload a PDF to get started.</p>
</div>""", unsafe_allow_html=True)
        st.stop()

    # ── Context strip ──────────────────────────────────────────────────────
    selected_scope = st.session_state.get("doc_scope", "All documents")
    context_docs = (
        st.session_state.documents if selected_scope == "All documents"
        else [d for d in st.session_state.documents if d["original_name"] == selected_scope]
    )
    doc_pills = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;'
        f'background:rgba(90,200,120,.07);color:#5ac878;'
        f'border:0.5px solid rgba(90,200,120,.2);border-radius:4px;'
        f'padding:2px 9px;font-size:10px;margin-right:5px;">'
        f'<span style="width:4px;height:4px;border-radius:50%;background:#5ac878;'
        f'display:inline-block;flex-shrink:0;"></span>'
        f'{d["original_name"][:32]}{"…" if len(d["original_name"]) > 32 else ""}</span>'
        for d in context_docs
    )
    st.markdown(
        f'<div style="padding:6px 0 12px;border-bottom:0.5px solid #1a1a1a;'
        f'margin-bottom:14px;display:flex;align-items:center;flex-wrap:wrap;gap:4px;">'
        f'{doc_pills}</div>',
        unsafe_allow_html=True,
    )

    # ── Empty state ────────────────────────────────────────────────────────
    if not messages:
        st.markdown("""
<div style="text-align:center;padding:48px 16px 36px;">
  <div style="position:relative;width:72px;height:72px;margin:0 auto 20px;display:inline-block;">
    <div style="position:absolute;inset:-14px;border-radius:50%;
                background:radial-gradient(circle,rgba(245,197,24,.22) 0%,transparent 70%);
                animation:glow 2.5s ease-in-out infinite;"></div>
    <div style="width:72px;height:72px;border-radius:50%;
                background:linear-gradient(135deg,rgba(245,197,24,.14),rgba(245,197,24,.04));
                border:0.5px solid rgba(245,197,24,.25);
                display:flex;align-items:center;justify-content:center;
                font-size:32px;position:relative;">💬</div>
  </div>
  <h3 style="font-size:17px;font-weight:500;color:#f0f0f0;margin:0 0 7px;">
    Ask anything about your contracts
  </h3>
  <p style="font-size:12px;color:#555;margin:0;line-height:1.7;">
    Extract clauses &nbsp;·&nbsp; Find obligations &nbsp;·&nbsp; Analyse risk &nbsp;·&nbsp; Compare terms
  </p>
</div>""", unsafe_allow_html=True)

    # ── Message history ────────────────────────────────────────────────────
    for msg_idx, msg in enumerate(messages):
        if msg.get("is_error"):
            st.error(msg["content"])
            continue
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑"):
            st.write(msg["content"])
            srcs = msg.get("sources", [])
            if srcs:
                chips = " ".join(
                    f'<span class="src-chip">📎 {s}</span>'
                    for s in list(dict.fromkeys(srcs))[:4]
                )
                st.markdown(f'<div style="margin-top:8px;">{chips}</div>', unsafe_allow_html=True)

            if msg["role"] == "assistant":
                st.markdown(_pipeline_metrics_html(msg), unsafe_allow_html=True)

                st.markdown('<div class="eval-btn">', unsafe_allow_html=True)
                if st.button(
                    "📊 Evaluate (RAGAS · GPT-4o mini)",
                    key=f"eval_{active_key}_{msg_idx}",
                ):
                    question_text = messages[msg_idx - 1]["content"]
                    contexts = [
                        c.get("content", "") for c in msg.get("chunks", []) if c.get("content")
                    ]
                    with st.spinner("Judging with GPT-4o mini…"):
                        try:
                            msg["ragas_score"] = score_turn(question_text, msg["content"], contexts)
                            msg["ragas_error"] = None
                        except Exception as exc:
                            msg["ragas_score"] = None
                            msg["ragas_error"] = str(exc)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

                if msg.get("ragas_error"):
                    st.error(f"RAGAS evaluation failed: {msg['ragas_error']}")

                if msg.get("ragas_score"):
                    st.markdown(_ragas_metrics_html(msg["ragas_score"]), unsafe_allow_html=True)
                    st.markdown(_ragas_chunks_html(msg.get("chunks", [])), unsafe_allow_html=True)

    # ── Chat input ─────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your contracts…"):
        session["messages"].append({"role": "user", "content": prompt})

        # Auto-title from first user message
        if session["title"] == "New chat":
            session["title"] = prompt[:45]

        with st.chat_message("assistant", avatar="🤖"):
            ph = st.empty()
            ph.markdown(_thinking_html(), unsafe_allow_html=True)
            try:
                payload = {
                    "query":      prompt,
                    "session_id": session["api_session_id"],
                }
                resp = requests.post(f"{API}/chat/", json=payload, timeout=60)
                ph.empty()
                if resp.ok:
                    data   = resp.json()
                    answer = data["answer"]
                    srcs   = data.get("sources", [])
                    session["api_session_id"] = data.get("session_id") or session["api_session_id"]
                    st.write(answer)
                    if srcs:
                        chips = " ".join(
                            f'<span class="src-chip">📎 {s}</span>'
                            for s in list(dict.fromkeys(srcs))[:4]
                        )
                        st.markdown(f'<div style="margin-top:8px;">{chips}</div>', unsafe_allow_html=True)
                    session["messages"].append(
                        {
                            "role":              "assistant",
                            "content":           answer,
                            "sources":           srcs,
                            "chunks":            data.get("chunks", []),
                            "intent_confidence": data.get("intent_confidence", 0.0),
                            "llm_latency_ms":    data.get("llm_latency_ms", 0.0),
                            "input_tokens":      data.get("input_tokens", 0),
                            "output_tokens":     data.get("output_tokens", 0),
                        }
                    )
                else:
                    try:
                        detail = resp.json().get("detail", resp.text[:300])
                    except Exception:
                        detail = resp.text[:300]
                    err = f"Error {resp.status_code}: {detail}"
                    st.error(err)
                    session["messages"].append({"role": "assistant", "content": err, "is_error": True})
            except requests.exceptions.ConnectionError:
                ph.empty()
                msg_text = "Cannot reach http://localhost:8000 — make sure `python main.py` is running."
                st.error(msg_text)
                session["messages"].append({"role": "assistant", "content": msg_text, "is_error": True})
            except Exception as exc:
                ph.empty()
                st.error(str(exc))
                session["messages"].append({"role": "assistant", "content": str(exc), "is_error": True})

        st.rerun()
