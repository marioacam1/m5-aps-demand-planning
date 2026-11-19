"""Ejecuta app.py con Streamlit y Altair simulados para detectar errores de Python (no de API visual)."""
import os, sys, types, shutil
from pathlib import Path
from unittest.mock import MagicMock
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"tests"))
BTN = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else set()

class Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, n): return getattr(st, n)
class St(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit"); self.session_state = {}; self.secrets = {}; self.calls = []
    def cache_data(self, *a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]
    def set_page_config(self, **k): pass
    def stop(self): raise SystemExit("stop")
    def tabs(self, names): return [Ctx() for _ in names]
    def columns(self, spec): return [Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
    def sidebar_(self): pass
    def spinner(self, *a, **k): return Ctx()
    def expander(self, *a, **k): return Ctx()
    def selectbox(self, label, options, index=0, format_func=str, key=None, **k):
        opts = list(options)
        if key in self.session_state and self.session_state[key] in opts: return self.session_state[key]
        [format_func(o) for o in opts[:5]]; return opts[min(index, len(opts)-1)]
    def number_input(self, label, value=0, **k): return value if k.get('key') not in self.session_state else self.session_state[k['key']]
    def text_input(self, label, value='', **k):
        key = k.get('key') or ''
        if os.environ.get('OVR'):
            if key.startswith("sal|") and key.endswith("|2"): return "+200"
            if key.startswith('mkt|') and key.endswith('|3'): return "500"
            if key.startswith('con_comment|'): return 'test comment'
        return self.session_state.get(key, value) if key else value
    def data_editor(self, df, **k):
        d = getattr(df, 'data', df)
        if hasattr(df, 'to_html'): df.to_html()
        return d.copy()
    def radio(self, label, options, index=0, key=None, **k):
        opts = list(options)
        if key in self.session_state and self.session_state[key] in opts: return self.session_state[key]
        if key in ('fc_view', 'con_view') and 'VIEW' in os.environ: return os.environ['VIEW']
        return opts[index]
    def container(self): return Ctx()
    def file_uploader(self, *a, **k): return None
    def rerun(self): pass
    def popover(self, *a, **k): return Ctx()
    def checkbox(self, label, value=False, key=None, **k): return self.session_state.get(key, value) if key else value
    def dataframe(self, obj, *a, **k):
        if hasattr(obj, 'to_html'): obj.to_html()      # force the Styler to compute every style function
        self.calls.append(('dataframe', ()))
    def multiselect(self, label, options=(), key=None, **k): return self.session_state.get(key, [])
    def slider(self, label, lo, hi, val=None, step=None, **k): return val if val is not None else lo
    def toggle(self, label, value=False, **k): return value
    def button(self, label, key=None, **k): return key in BTN
    def download_button(self, *a, **k): return False
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        def f(*a, **k): self.calls.append((n, a[:1])); return MagicMock()
        return f
st = St(); st.session_state.update(__import__("json").loads(os.environ.get("SEED","{}"))); 
if os.environ.get("COMMIT"):
    lab = "All categories · All stores · All departments"
    st.session_state["committed"] = {lab: {278: {"final": 60000.0, "sales": "+10", "mkt": "", "reason": "Promotion", "comment": "x", "user": "u", "ts": "t"}}}
    st.session_state["versions"] = [{"name": "V1", "scope": lab, "ts": "t", "user": "u", "total": 1.0, "weeks": {w: 50000.0 for w in range(278, 291)}}]
st.sidebar = Ctx(); st.column_config = MagicMock()
sys.modules["streamlit"] = st
_comp = types.ModuleType("streamlit.components"); _v1 = types.ModuleType("streamlit.components.v1"); _v1.html = lambda *a, **k: st.calls.append(("html", ()))
_comp.v1 = _v1; st.components = _comp
sys.modules["streamlit.components"] = _comp; sys.modules["streamlit.components.v1"] = _v1
sys.modules["altair"] = MagicMock()
import fake_endpoint as fe
srv, host = fe.start()
os.environ.update(DATABRICKS_HOST=host, DATABRICKS_TOKEN="good-token", ENDPOINT_NAME="m5-aps-forecast")
import data_access; data_access.DATA_DIR = Path(sys.argv[1]); data_access.load_all.__defaults__ = (Path(sys.argv[1]),)
src = (ROOT/"app.py").read_text()
exec(compile(src, "app.py", "exec"), {"__name__": "__main__", "__file__": str(ROOT/"app.py")})
errs = [c for c in st.calls if c[0] in ("error", "warning")]
print("llamadas st:", len(st.calls), "| errores/avisos:", errs)
print("session:", {k: type(v).__name__ for k, v in st.session_state.items()})
