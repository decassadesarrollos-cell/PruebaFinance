"""
MarginEdge — Dashboard de daytrading gratuito
Almacenamiento: Google Sheets (historial permanente, accesible desde cualquier dispositivo)
Datos de mercado: yfinance

Configuración de Sheets en Streamlit Cloud:
  Settings → Secrets → pegar el contenido de .streamlit/secrets.toml

Fallback: si no hay secrets configurados, usa CSV local (para desarrollo).
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import os, io, time, json
from datetime import datetime, timezone, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="MarginEdge",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#06080b}
[data-testid="stSidebar"]{background:#0a0d12;border-right:1px solid #131c28}
[data-testid="stHeader"]{background:#06080b}
div[data-testid="metric-container"]{
  background:#0d1520;border:1px solid #1a2535;border-radius:8px;padding:10px 12px}
[data-testid="metric-container"] label{color:#4a6070!important;font-size:10px!important}
[data-testid="metric-container"] [data-testid="metric-value"]{
  color:#dde4f0!important;font-size:18px!important}
.stButton>button{background:#0d1520!important;border:1px solid #1a2535!important;
  color:#dde4f0!important;border-radius:6px!important;font-size:11px!important}
.stButton>button:hover{border-color:#00e5a0!important;color:#00e5a0!important}
.stTabs [data-baseweb="tab-list"]{background:#0a0d12;border-bottom:1px solid #131c28}
.stTabs [data-baseweb="tab"]{color:#4a6070!important;font-size:12px!important}
.stTabs [aria-selected="true"]{color:#dde4f0!important;border-bottom-color:#00e5a0!important}
label[data-testid="stWidgetLabel"]{color:#4a6070!important;font-size:10px!important;
  text-transform:uppercase;letter-spacing:.4px}
h1,h2,h3{color:#dde4f0!important}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  HORA CDMX
# ═══════════════════════════════════════════════════════════
def cdmx_now():
    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=6)

def dst_usa():
    n = cdmx_now(); y = n.year
    s = datetime(y,3,1)  + timedelta(days=(6-datetime(y,3,1).weekday()+7)%7+7)
    e = datetime(y,11,1) + timedelta(days=(6-datetime(y,11,1).weekday())%7)
    return s <= n < e

DST         = dst_usa()
MKT_OPEN_H  = 7  if DST else 8
MKT_CLOSE_H = 14 if DST else 15
OPEN_STR    = "7:30 AM"  if DST else "8:30 AM"
CLOSE_STR   = "2:00 PM"  if DST else "3:00 PM"
NOW         = cdmx_now()
H_NOW       = NOW.hour + NOW.minute / 60
MKT_LIVE    = (MKT_OPEN_H + 0.5) <= H_NOW < MKT_CLOSE_H

# ═══════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════
SHEET_NAME  = "MarginEdge_Trades"
TAB_TRADES  = "trades"
FALLBACK_CSV= "marginedge_trades_local.csv"
ATR_MULT    = 1.0

TRADE_COLS = [
    "id","fecha","hora","ticker","direction",
    "entry","sl","tp1","shares",
    "exit_price","exit_hora","resultado",
    "pnl_usd","pnl_pct","score","signals","notas",
]

# ── Trades de demostración ───────────────────────────────
def _make_demo_trades():
    """Datos falsos para probar la app sin tocar Sheets."""
    hoy = (datetime.now(timezone.utc)-timedelta(hours=6)).strftime("%Y-%m-%d")
    ayer = (datetime.now(timezone.utc)-timedelta(hours=6)-timedelta(days=1)).strftime("%Y-%m-%d")
    ant  = (datetime.now(timezone.utc)-timedelta(hours=6)-timedelta(days=2)).strftime("%Y-%m-%d")
    return pd.DataFrame([
        {"id":"d1","fecha":hoy,  "hora":"08:15","ticker":"NVDA","direction":"long",
         "entry":177.50,"sl":175.80,"tp1":181.90,"shares":8,
         "exit_price":181.90,"exit_hora":"09:42","resultado":"win",
         "pnl_usd":27.20,"pnl_pct":1.53,"score":3,"signals":"VWAP+EMA","notas":"demo"},
        {"id":"d2","fecha":hoy,  "hora":"09:05","ticker":"AMD","direction":"long",
         "entry":122.30,"sl":120.90,"tp1":125.10,"shares":10,
         "exit_price":120.90,"exit_hora":"09:55","resultado":"loss",
         "pnl_usd":-14.00,"pnl_pct":-1.14,"score":2,"signals":"EMA","notas":"demo"},
        {"id":"d3","fecha":hoy,  "hora":"09:50","ticker":"TSLA","direction":"short",
         "entry":245.60,"sl":247.80,"tp1":241.20,"shares":6,
         "exit_price":241.20,"exit_hora":"11:10","resultado":"win",
         "pnl_usd":26.40,"pnl_pct":1.79,"score":2,"signals":"VWAP+BRK","notas":"demo"},
        {"id":"d4","fecha":hoy,  "hora":"12:20","ticker":"AAPL","direction":"long",
         "entry":213.40,"sl":211.90,"tp1":216.40,"shares":9,
         "exit_price":None,"exit_hora":None,"resultado":None,
         "pnl_usd":None,"pnl_pct":None,"score":2,"signals":"VWAP","notas":"demo — abierta"},
        {"id":"d5","fecha":ayer, "hora":"08:30","ticker":"MSFT","direction":"long",
         "entry":415.20,"sl":412.50,"tp1":420.60,"shares":3,
         "exit_price":420.60,"exit_hora":"10:15","resultado":"win",
         "pnl_usd":16.20,"pnl_pct":1.30,"score":3,"signals":"VWAP+EMA+BRK","notas":"demo"},
        {"id":"d6","fecha":ayer, "hora":"11:45","ticker":"META","direction":"short",
         "entry":582.10,"sl":585.40,"tp1":575.50,"shares":2,
         "exit_price":575.50,"exit_hora":"13:20","resultado":"win",
         "pnl_usd":13.20,"pnl_pct":1.13,"score":2,"signals":"EMA+BRK","notas":"demo"},
        {"id":"d7","fecha":ant,  "hora":"09:10","ticker":"NVDA","direction":"short",
         "entry":168.90,"sl":171.20,"tp1":164.50,"shares":8,
         "exit_price":171.20,"exit_hora":"09:50","resultado":"loss",
         "pnl_usd":-18.40,"pnl_pct":-1.36,"score":1,"signals":"VWAP","notas":"demo"},
        {"id":"d8","fecha":ant,  "hora":"12:05","ticker":"AMD","direction":"long",
         "entry":119.80,"sl":118.20,"tp1":123.00,"shares":11,
         "exit_price":123.00,"exit_hora":"13:30","resultado":"win",
         "pnl_usd":35.20,"pnl_pct":2.67,"score":3,"signals":"VWAP+EMA+BRK","notas":"demo"},
    ])

# ═══════════════════════════════════════════════════════════
#  GOOGLE SHEETS — conexión
#  Usa gspread 5.x (API estable, sin cambios breaking)
# ═══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_sheet():
    """
    Conecta con Google Sheets vía service account.
    Devuelve (worksheet, modo) donde modo es 'sheets' o 'csv'.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        # Leer service account desde Streamlit secrets
        raw = st.secrets.get("GOOGLE_SERVICE_ACCOUNT", None)
        if raw is None:
            return None, "csv"

        creds_dict = json.loads(raw) if isinstance(raw, str) else dict(raw)
        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds  = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)

        # Abrir o crear el spreadsheet
        try:
            sh = client.open(SHEET_NAME)
        except gspread.SpreadsheetNotFound:
            sh = client.create(SHEET_NAME)
            # Compartir con el email del service account (para que sea visible en Drive)
            sh.share(creds_dict.get("client_email",""), perm_type="user", role="writer")

        # Abrir o crear la hoja de trades
        try:
            ws = sh.worksheet(TAB_TRADES)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=TAB_TRADES, rows=2000, cols=len(TRADE_COLS))
            ws.append_row(TRADE_COLS)   # encabezado

        return ws, "sheets"

    except Exception as e:
        st.warning(f"Google Sheets no disponible ({e}). Usando CSV local.", icon="⚠️")
        return None, "csv"

# ═══════════════════════════════════════════════════════════
#  TRADES — lectura/escritura (Sheets o CSV)
# ═══════════════════════════════════════════════════════════
def numify(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_trades() -> pd.DataFrame:
    # Modo demo — devuelve datos falsos sin tocar Sheets
    if st.session_state.get("demo_mode", False):
        df = _make_demo_trades()
        for c in TRADE_COLS:
            if c not in df.columns: df[c] = None
        return df[TRADE_COLS].copy()

    ws, mode = get_sheet()

    if mode == "sheets" and ws is not None:
        try:
            data = ws.get_all_values()
            if len(data) <= 1:
                return pd.DataFrame(columns=TRADE_COLS)
            df = pd.DataFrame(data[1:], columns=data[0])
            # Asegurar que tiene todas las columnas
            for c in TRADE_COLS:
                if c not in df.columns:
                    df[c] = None
            # Reemplazar strings vacíos por None
            df = df.replace("", None)
            return df[TRADE_COLS].copy()
        except Exception as e:
            st.error(f"Error leyendo Sheets: {e}")
            return pd.DataFrame(columns=TRADE_COLS)

    # Fallback CSV
    if os.path.exists(FALLBACK_CSV):
        try:
            df = pd.read_csv(FALLBACK_CSV, dtype=str)
            for c in TRADE_COLS:
                if c not in df.columns:
                    df[c] = None
            return df[TRADE_COLS].copy()
        except Exception:
            pass
    return pd.DataFrame(columns=TRADE_COLS)

def _save_to_sheets(df: pd.DataFrame):
    """Reescribe la hoja completa. Óptimo para <500 trades."""
    ws, mode = get_sheet()
    if mode != "sheets" or ws is None:
        return False
    try:
        ws.clear()
        rows = [TRADE_COLS] + df[TRADE_COLS].fillna("").astype(str).values.tolist()
        ws.update(rows, value_input_option="RAW")
        return True
    except Exception as e:
        st.error(f"Error guardando en Sheets: {e}")
        return False

def _save_csv(df: pd.DataFrame):
    df.to_csv(FALLBACK_CSV, index=False)

def save_trades(df: pd.DataFrame):
    # Modo demo — no guarda nada
    if st.session_state.get("demo_mode", False):
        return
    _, mode = get_sheet()
    if mode == "sheets":
        if not _save_to_sheets(df):
            _save_csv(df)   # fallback a CSV si falla Sheets
    else:
        _save_csv(df)

def add_trade(capital, risk_pct, rr,
              ticker, direction, entry, sl, tp1,
              score, signals, notas) -> int:
    df   = load_trades()
    risk = abs(entry - sl) if abs(entry - sl) > 0 else entry * risk_pct / 100
    sh   = max(1, int(capital * risk_pct / 100 / risk))
    if not tp1 or tp1 == 0:
        tp1 = round(
            entry + abs(entry - sl) * rr if direction == "long"
            else entry - abs(entry - sl) * rr, 2
        )
    n   = cdmx_now()
    row = {c: "" for c in TRADE_COLS}
    row.update({
        "id"       : str(int(time.time() * 1000)),
        "fecha"    : n.strftime("%Y-%m-%d"),
        "hora"     : n.strftime("%H:%M"),
        "ticker"   : ticker.upper(),
        "direction": direction,
        "entry"    : entry,
        "sl"       : sl,
        "tp1"      : tp1,
        "shares"   : sh,
        "score"    : score,
        "signals"  : signals,
        "notas"    : notas,
    })
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_trades(df)
    return sh

def close_trade(trade_id, exit_price, capital, risk_pct, nota=""):
    df   = load_trades()
    mask = df["id"].astype(str) == str(trade_id)
    if not mask.any():
        return None
    df   = numify(df, ["entry","sl","shares"])
    idx  = df[mask].index[0]
    entry  = float(df.loc[idx, "entry"])
    dirn   = str(df.loc[idx, "direction"])
    sl_val = float(df.loc[idx, "sl"])
    sh     = float(df.loc[idx, "shares"]) if pd.notna(df.loc[idx, "shares"]) \
             else max(1, int(capital * risk_pct / 100 / max(abs(entry - sl_val), 0.01)))
    diff   = (exit_price - entry) if dirn == "long" else (entry - exit_price)
    pnl    = round(diff * sh, 2)
    n      = cdmx_now()
    df.loc[idx, "exit_price"] = exit_price
    df.loc[idx, "exit_hora"]  = n.strftime("%H:%M")
    df.loc[idx, "pnl_usd"]   = pnl
    df.loc[idx, "pnl_pct"]   = round(diff / entry * 100, 3)
    df.loc[idx, "resultado"]  = "win" if pnl > 0 else "loss"
    df.loc[idx, "notas"]      = nota or df.loc[idx, "notas"]
    save_trades(df)
    return pnl

# ═══════════════════════════════════════════════════════════
#  INDICADORES
# ═══════════════════════════════════════════════════════════
def compute_indicators(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    for span, name in [(9,"e9"),(21,"e21"),(50,"e50")]:
        d[name] = d["Close"].ewm(span=span, adjust=False).mean()
    hlc3 = (d["High"] + d["Low"] + d["Close"]) / 3

    # Bug fix: yfinance devuelve índice tz-aware (US/Eastern).
    # .date en un índice tz-aware de pandas retorna la fecha LOCAL correcta,
    # pero en algunas versiones falla o agrupa mal. Normalizamos explícitamente.
    try:
        if hasattr(d.index, "tz") and d.index.tz is not None:
            date_key = d.index.tz_convert("America/New_York").date
        else:
            date_key = d.index.date
        date_ser = pd.Series(date_key, index=d.index)
        d["vwap"] = (
            (hlc3 * d["Volume"]).groupby(date_ser).cumsum() /
            d["Volume"].groupby(date_ser).cumsum()
        )
    except Exception:
        # Fallback: VWAP rolling 20 barras si falla el groupby por fecha
        vol_sum = d["Volume"].rolling(20, min_periods=1).sum()
        d["vwap"] = (hlc3 * d["Volume"]).rolling(20, min_periods=1).sum() / vol_sum.replace(0, np.nan)
    dx = d["Close"].diff()
    g  = dx.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    l  = (-dx.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["rsi"]  = 100 - (100 / (1 + g / l.replace(0, np.nan)))
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift()).abs(),
        (d["Low"]  - d["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    d["atr"]  = tr.ewm(alpha=1/14, adjust=False).mean()
    d["vavg"] = d["Volume"].rolling(20).mean()
    v1  = d["Volume"] > d["vavg"] * 1.5
    v2  = d["Volume"] > d["vavg"] * 2.0
    al  = (d["Close"] > d["e50"]) & (d["e21"] > d["e50"])
    bj  = (d["Close"] < d["e50"]) & (d["e21"] < d["e50"])
    rl  = d["rsi"] < 70
    rs  = d["rsi"] > 30
    xo  = lambda a,b: (a > b) & (a.shift() <= b.shift())
    xu  = lambda a,b: (a < b) & (a.shift() >= b.shift())
    s1l = xo(d["Close"], d["vwap"]) & v1 & al & rl
    s1s = xu(d["Close"], d["vwap"]) & v1 & bj & rs
    tl  = (d["Low"]  <= d["e21"] * 1.003) & (d["Close"] > d["e21"])
    ts  = (d["High"] >= d["e21"] * 0.997) & (d["Close"] < d["e21"])
    # s2l/s2s: toque de EMA21 funciona en mercado lateral —
    # solo requiere que precio esté del lado correcto de EMA21, no tendencia fuerte vs EMA50
    s2l = tl & ~tl.shift(fill_value=False) & (d["Close"] > d["e21"]) & rl
    s2s = ts & ~ts.shift(fill_value=False) & (d["Close"] < d["e21"]) & rs
    ph  = d["High"].shift().rolling(20).max()
    pl  = d["Low"].shift().rolling(20).min()
    s3l = xo(d["Close"], ph) & v2 & (d["Close"] > d["e21"]) & rl
    s3s = xu(d["Close"], pl) & v2 & (d["Close"] < d["e21"]) & rs
    d["scl"] = s1l.astype(int) + s2l.astype(int) + s3l.astype(int)
    d["scs"] = s1s.astype(int) + s2s.astype(int) + s3s.astype(int)
    return d

# ═══════════════════════════════════════════════════════════
#  ML
# ═══════════════════════════════════════════════════════════
def get_prob(df, direction, trades_df, ticker, rr) -> float:
    sc_col = "scl" if direction == "long" else "scs"
    rows   = []
    for i in range(40, len(df) - 16):
        if df[sc_col].iloc[i] < 2:
            continue
        r = df.iloc[i]
        if pd.isna(r["rsi"]) or pd.isna(r["atr"]):
            continue
        e = df["Open"].iloc[i + 1]
        if e <= 0 or pd.isna(e):
            continue
        sl = (df["Low"].iloc[max(0,i-2):i+1].min() - r["atr"] * ATR_MULT) \
             if direction == "long" \
             else (df["High"].iloc[max(0,i-2):i+1].max() + r["atr"] * ATR_MULT)
        tp  = e + (e-sl)*rr if direction=="long" else e - (sl-e)*rr
        feat = [
            float(r["rsi"]),
            r["atr"] / e,
            (r["Close"]-r["vwap"])/r["vwap"]*100 if r["vwap"]>0 else 0,
            (r["Close"]-r["e21"])/r["e21"]*100,
            (r["Close"]-r["e50"])/r["e50"]*100,
            min(r["Volume"]/r["vavg"] if r["vavg"]>0 else 1, 10),
            int(df[sc_col].iloc[i]),
            (r["Close"]/df["Close"].iloc[i-1]-1)*100 if i>1 else 0,
        ]
        lbl = 0
        for j in range(i+1, min(i+16, len(df))):
            b = df.iloc[j]
            if direction == "long":
                if b["Low"] <= sl:  lbl=0; break
                if b["High"] >= tp: lbl=1; break
            else:
                if b["High"] >= sl: lbl=0; break
                if b["Low"] <= tp:  lbl=1; break
        rows.append((feat, lbl, 1.0))

    if not trades_df.empty:
        td = numify(trades_df.copy(), ["rsi","vol_ratio","score"])
        td = td[
            (td["ticker"] == ticker.upper()) &
            (td["exit_price"].notna()) &
            (td["direction"] == direction)
        ]
        for _, t in td.iterrows():
            try:
                feat_r = [
                    float(t["rsi"])       if pd.notna(t.get("rsi"))       else 50.0,
                    0.01, 0.0, 0.0, 0.0,
                    float(t.get("vol_ratio",1.5)) if pd.notna(t.get("vol_ratio")) else 1.5,
                    float(t["score"])     if pd.notna(t.get("score"))     else 2.0,
                    0.0,
                ]
                rows.append((feat_r, 1 if str(t.get("resultado","loss"))=="win" else 0, 3.0))
            except Exception:
                pass

    if len(rows) < 12:
        return 50.0
    X = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=int)
    w = np.array([r[2] for r in rows], dtype=float)
    mask = np.isfinite(X).all(axis=1)
    X, y, w = X[mask], y[mask], w[mask]
    if len(X) < 12 or len(np.unique(y)) < 2:
        return 50.0
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    model  = RandomForestClassifier(
        n_estimators=80, max_depth=5,
        min_samples_leaf=max(2, len(X)//20),
        class_weight="balanced", random_state=42, n_jobs=1,
    )
    model.fit(Xs, y, sample_weight=w)
    fn = [
        float(df["rsi"].iloc[-1]),
        df["atr"].iloc[-1]/df["Close"].iloc[-1],
        (df["Close"].iloc[-1]-df["vwap"].iloc[-1])/df["vwap"].iloc[-1]*100
        if df["vwap"].iloc[-1]>0 else 0,
        (df["Close"].iloc[-1]-df["e21"].iloc[-1])/df["e21"].iloc[-1]*100,
        (df["Close"].iloc[-1]-df["e50"].iloc[-1])/df["e50"].iloc[-1]*100,
        min(df["Volume"].iloc[-1]/df["vavg"].iloc[-1]
            if df["vavg"].iloc[-1]>0 else 1, 10),
        int(df[sc_col].iloc[-1]),
        (df["Close"].iloc[-1]/df["Close"].iloc[-2]-1)*100 if len(df)>1 else 0,
    ]
    if not all(np.isfinite(fn)):
        return 50.0
    return round(float(model.predict_proba(scaler.transform([fn]))[0][1])*100, 1)

# ═══════════════════════════════════════════════════════════
#  SCANNER
# ═══════════════════════════════════════════════════════════
@st.cache_data(ttl=600, show_spinner=False)
def fetch_ticker(ticker, interval, _cache_key):
    """
    Descarga datos de yfinance con manejo robusto de errores.
    Devuelve (df_con_indicadores, error_msg) — error_msg es "" si todo OK.
    """
    try:
        # Bug fix: "10d" en lugar de "5d" para garantizar ≥30 barras
        # incluso en semanas con festivos o cuando el mercado cerró antes
        raw = yf.download(
            ticker, period="10d", interval=interval,
            auto_adjust=True, progress=False,
            timeout=20,          # Bug fix: 8s era demasiado corto en Streamlit Cloud
            prepost=False,       # excluir pre/post market (volumen 0 rompe VWAP)
        )

        # Bug fix: yfinance ≥0.2.50 siempre devuelve MultiIndex aunque sea 1 ticker
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.loc[:, ~raw.columns.duplicated()]   # eliminar columnas duplicadas

        # Validar columnas requeridas
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in raw.columns]
        if missing:
            return None, f"Columnas faltantes: {missing}"

        # Filtrar barras con volumen 0 (pre/post market y días sin datos)
        raw = raw[raw["Volume"] > 0].dropna(subset=["Close", "High", "Low"]).copy()

        # Bug fix: umbral reducido a 30 (antes 60); con "10d" siempre hay suficientes
        if raw.empty or len(raw) < 30:
            return None, f"Datos insuficientes: {len(raw)} barras"

        return compute_indicators(raw), ""

    except Exception as e:
        return None, str(e)

def analyze_ticker(df, ticker, capital, risk_pct, rr):
    if df is None:
        return None
    am = rr / 4.0
    # Fix: 15 barras (3.75 h en 15m) en lugar de 5 (75 min)
    # Esto garantiza que señales de la primera ventana del día sigan visibles
    for offset in range(15):
        idx = -(1 + offset)
        try:
            row = df.iloc[idx]
        except Exception:
            continue
        if pd.isna(row["rsi"]) or pd.isna(row["atr"]):
            continue
        scl   = int(row["scl"])
        scs   = int(row["scs"])
        score = max(scl, scs)
        if score < 1:
            continue
        dirn  = "long" if scl >= scs else "short"
        vavg  = row["vavg"] if row["vavg"] > 0 else 1
        vr    = row["Volume"] / vavg
        # Fix: umbral consistente 0.5x para todos los offsets.
        # Antes era 1.0x para offset>0, lo que descartaba señales de EMA (s2l)
        # que no requieren volumen alto pero sí se generaban hace 2-3 barras.
        if vr < 0.5:
            continue
        if dirn == "long"  and row["rsi"] > 73: continue
        if dirn == "short" and row["rsi"] < 27: continue
        px  = float(row["Close"])
        sl  = (px - row["atr"]*am) if dirn=="long" else (px + row["atr"]*am)
        tp  = px + (px-sl)*rr      if dirn=="long" else px - (sl-px)*rr
        rps = abs(px - sl)
        sh  = max(1, int(capital * risk_pct/100 / rps)) if rps > 0 else 1
        active = []
        pv = df["Close"].iloc[idx-1] if abs(idx)>1 else px
        if dirn == "long":
            if px>row["vwap"] and pv<=row["vwap"]: active.append("VWAP")
            if row["Low"]<=row["e21"]*1.003:        active.append("EMA")
            ph = df["High"].shift().rolling(20).max().iloc[idx]
            if not pd.isna(ph) and px>ph:           active.append("BRK")
        else:
            if px<row["vwap"] and pv>=row["vwap"]: active.append("VWAP")
            if row["High"]>=row["e21"]*0.997:       active.append("EMA")
            pl = df["Low"].shift().rolling(20).min().iloc[idx]
            if not pd.isna(pl) and px<pl:           active.append("BRK")
        return {
            "ticker"   : ticker,   "direction": dirn,     "score"  : score,
            "price"    : round(px,2),   "sl"    : round(sl,2),
            "tp"       : round(tp,2),   "sl_pct": round(abs(sl-px)/px*100,2),
            "tp_pct"   : round(abs(tp-px)/px*100,2),
            "shares"   : sh,            "cap_pct": round(sh*px/capital*100,1),
            "vol"      : round(vr,1),   "rsi"   : round(float(row["rsi"]),1),
            "signals"  : "+".join(active) if active else f"s{score}",
            "hora"     : str(df.index[idx])[11:16],
        }
    return None

def run_scan(tickers, interval, capital, risk_pct, rr, trades_df):
    cache_key = int(time.time() // 600)
    results   = []
    diag      = []   # diagnóstico por ticker para debug
    prog      = st.progress(0, text="Iniciando...")
    for i, tk in enumerate(tickers):
        prog.progress((i+1)/len(tickers), text=f"Analizando {tk}... ({i+1}/{len(tickers)})")
        df_result = fetch_ticker(tk, interval, cache_key)
        # fetch_ticker ahora devuelve (df, error_msg)
        if isinstance(df_result, tuple):
            df, err = df_result
        else:
            # compatibilidad si el caché aún tiene el formato antiguo
            df, err = df_result, ""
        if df is None:
            diag.append({"tk": tk, "ok": False, "msg": err or "Sin datos suficientes"})
            continue
        r = analyze_ticker(df, tk, capital, risk_pct, rr)
        if r is None:
            # Diagnóstico detallado: muestra las últimas 15 barras revisadas
            bar_details = []
            for off in range(min(15, len(df))):
                try:
                    b = df.iloc[-(1+off)]
                    bscl = int(b["scl"]) if not pd.isna(b.get("scl",0)) else 0
                    bscs = int(b["scs"]) if not pd.isna(b.get("scs",0)) else 0
                    bvr  = round(b["Volume"]/b["vavg"],1) if b.get("vavg",0)>0 else 0
                    brs  = round(float(b.get("rsi",0)),0)
                    if max(bscl,bscs) > 0:
                        bar_details.append(f"offset-{off}: scl={bscl} scs={bscs} vol={bvr}x rsi={brs} ← señal pero filtrada")
                except Exception:
                    pass
            detail_msg = " | ".join(bar_details) if bar_details else "scl=0 scs=0 en las 15 barras revisadas"
            diag.append({
                "tk": tk, "ok": False,
                "msg": detail_msg,
            })
            continue
        diag.append({"tk": tk, "ok": True, "msg": f"✅ score={r['score']} {r['direction'].upper()}"})
        try:
            r["prob"] = get_prob(df, r["direction"], trades_df, tk, rr)
        except Exception:
            r["prob"] = 50.0
        results.append(r)
    prog.empty()
    # Guardar diagnóstico en session_state para mostrarlo en la UI
    st.session_state["scan_diag"] = diag
    return sorted(results, key=lambda x: (x["score"], x["prob"]), reverse=True)

# ═══════════════════════════════════════════════════════════
#  ESTADÍSTICAS
# ═══════════════════════════════════════════════════════════
def calc_stats(cl: pd.DataFrame) -> dict:
    if cl.empty:
        return {}
    cl       = numify(cl.copy(), ["pnl_usd","entry","exit_price","score"])
    pnl_all  = cl["pnl_usd"].fillna(0)
    wr       = (cl["resultado"] == "win").mean() * 100
    wins_sum = cl[pnl_all > 0]["pnl_usd"].sum()
    loss_sum = abs(cl[pnl_all < 0]["pnl_usd"].sum())
    def by(col):
        return (
            cl.groupby(col)
            .agg(
                n=("resultado","count"),
                wr=("resultado", lambda x:(x=="win").mean()*100),
                pnl=("pnl_usd","sum"),
            )
            .round(1).reset_index()
        )
    s = {
        "n" : len(cl), "wr": round(wr,1),
        "pnl": round(pnl_all.sum(),2),
        "pf" : round(wins_sum/loss_sum,2) if loss_sum>0 else 0,
        "by_ticker": by("ticker").sort_values("pnl",ascending=False),
        "by_score" : by("score").sort_values("score"),
    }
    if "hora" in cl.columns:
        cl = cl.copy()
        cl["hh"] = cl["hora"].astype(str).str[:2].fillna("09")
        s["by_hora"] = by("hh").sort_values("pnl",ascending=False)
    return s

# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ MarginEdge")

    # ── Modo demo ──────────────────────────────────────
    demo_mode = st.toggle("🧪 Modo demo", value=st.session_state.get("demo_mode", False),
                          help="Usa datos de prueba — no lee ni escribe en Google Sheets")
    st.session_state["demo_mode"] = demo_mode
    if demo_mode:
        st.warning("🧪 DEMO ACTIVO · Sheets no se toca", icon="🧪")
    else:
        # Badge modo almacenamiento real
        _, mode = get_sheet()
        if mode == "sheets":
            st.success("☁ Google Sheets activo")
        else:
            st.info("💾 Modo CSV local")

    if MKT_LIVE:
        st.success(f"🟢 ABIERTO · {OPEN_STR}–{CLOSE_STR}")
    else:
        st.warning(f"🟡 CERRADO · Abre {OPEN_STR}")
    st.caption(f"{'DST activo' if DST else 'Hora estándar'} · {NOW.strftime('%H:%M')} CDMX")
    st.divider()

    st.markdown("### Configuración")
    capital   = st.number_input("Capital ($)",   value=10000.0, step=500.0, format="%.0f")
    risk_pct  = st.slider("Riesgo/op (%)",       0.5, 3.0, 1.0, 0.1)
    rr_ratio  = st.slider("R:R ratio",           1.0, 5.0, 2.0, 0.5)
    stop_pct  = st.slider("Stop diario (%)",     1.0, 6.0, 3.0, 0.5)
    interval  = st.selectbox("Timeframe",        ["1m","5m","15m","1h"], index=2)
    min_score = st.selectbox("Score mínimo",     [1,2,3], index=0,
                              format_func=lambda x: f"{x}/3")
    min_prob  = st.slider("Prob ML mínima (%)",  0, 100, 50, 5)
    st.divider()

    st.markdown("### Tickers")
    tickers_raw = st.text_area(
        "Uno por línea",
        value="\n".join(["NVDA","AAPL","MSFT","TSLA","AMD","META","AMZN","GOOGL"]),
        height=160,
    )
    tickers = [t.strip().upper() for t in tickers_raw.split("\n") if t.strip()]
    st.divider()

    if st.button("🔄 Actualizar señales", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.session_state.pop("scan_results", None)
        st.session_state.pop("scan_key", None)
        st.rerun()

# ═══════════════════════════════════════════════════════════
#  CARGAR DATOS
# ═══════════════════════════════════════════════════════════
trades_all = load_trades()
trades_num = numify(
    trades_all.copy(),
    ["entry","sl","tp1","exit_price","pnl_usd","pnl_pct","shares","score"],
)
hoy          = NOW.strftime("%Y-%m-%d")
stop_limite  = capital * stop_pct / 100
t_hoy        = trades_num[trades_num["fecha"]==hoy].copy() \
               if not trades_num.empty else pd.DataFrame(columns=TRADE_COLS)
cerr_h       = t_hoy[t_hoy["exit_price"].notna()]  if not t_hoy.empty else pd.DataFrame()
abrt_h       = t_hoy[t_hoy["exit_price"].isna()]   if not t_hoy.empty else pd.DataFrame()
pnl_hoy      = cerr_h["pnl_usd"].sum() if len(cerr_h) else 0.0
wr_hoy       = (cerr_h["resultado"]=="win").mean()*100 if len(cerr_h) else 0.0
stop_pct_uso = min(abs(pnl_hoy)/stop_limite*100 if pnl_hoy<0 else 0, 100)
stop_color   = "#ff3d55" if stop_pct_uso>75 else "#ffb020" if stop_pct_uso>40 else "#00e5a0"

# ═══════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════
tab_sen, tab_hoy, tab_his, tab_calc = st.tabs([
    "⚡ Señales","📅 Hoy","📈 Historial","🔢 Calculadora",
])

# ─── TAB 1 — SEÑALES ────────────────────────────────────
with tab_sen:
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Capital",     f"${capital:,.0f}")
    c2.metric("Riesgo/op",   f"{risk_pct}%", f"${capital*risk_pct/100:.0f}")
    c3.metric("P&L hoy",     f"${pnl_hoy:+.0f}")
    c4.metric("Stop diario", f"${stop_limite:.0f}",
              f"Usado {stop_pct_uso:.0f}%",
              delta_color="inverse" if stop_pct_uso>50 else "normal")
    c5.metric("Ops hoy", len(t_hoy), f"{len(abrt_h)} abiertas")

    # Barra stop
    restante = max(0, stop_limite + pnl_hoy)
    st.markdown(f"""
    <div style="background:#0d1520;border:1px solid #1a2535;border-radius:8px;
                padding:10px 14px;margin:8px 0">
      <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:5px">
        <span style="color:#4a6070">Stop diario ${stop_limite:.0f} ({stop_pct}%)</span>
        <span style="color:{stop_color}">
          Usado ${abs(min(pnl_hoy,0)):.0f} · Restante ${restante:.0f}
        </span>
      </div>
      <div style="background:#131c28;border-radius:4px;height:6px;overflow:hidden">
        <div style="width:{stop_pct_uso}%;height:100%;background:{stop_color};border-radius:4px"></div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # Scanner con cache
    scan_key = f"{'-'.join(sorted(tickers))}_{interval}_{int(time.time()//600)}"
    if st.session_state.get("scan_key") != scan_key:
        with st.spinner(f"Analizando {len(tickers)} tickers..."):
            results = run_scan(tickers, interval, capital, risk_pct, rr_ratio, trades_num)
        st.session_state["scan_results"] = results
        st.session_state["scan_key"]     = scan_key

    all_sigs = st.session_state.get("scan_results", [])
    signals  = [s for s in all_sigs
                if s["score"] >= min_score and s["prob"] >= min_prob]

    r1, r2 = st.columns([1,5])
    with r2:
        longs  = sum(1 for s in signals if s["direction"]=="long")
        shorts = sum(1 for s in signals if s["direction"]=="short")
        st.caption(
            f"{len(signals)} señal(es) · {longs} LONG · {shorts} SHORT · "
            f"Actualizado {NOW.strftime('%H:%M')}"
        )

    if not signals:
        st.info("Sin señales con esos filtros. Baja el score mínimo o presiona '🔄 Actualizar señales'.")

    # ── Panel de diagnóstico ─────────────────────────────────
    diag = st.session_state.get("scan_diag", [])
    if diag:
        ok_n  = sum(1 for d in diag if d["ok"])
        err_n = len(diag) - ok_n
        lbl   = f"🔍 Diagnóstico del scanner — {ok_n} señal(es) · {err_n} sin señal"
        with st.expander(lbl, expanded=(ok_n == 0 and err_n > 0)):
            for d in diag:
                icon  = "🟢" if d["ok"] else "🔴"
                color = "#00e5a0" if d["ok"] else "#4a6070"
                st.markdown(
                    f"<span style='font-size:11px;font-family:monospace'>"
                    f"{icon} <b style='color:#dde4f0'>{d['tk']:<6}</b> "
                    f"<span style='color:{color}'>{d['msg']}</span></span>",
                    unsafe_allow_html=True,
                )

    if not signals:
        pass
    else:
        cols = st.columns(min(len(signals), 4))
        for i, s in enumerate(signals):
            sc_c  = "#00e5a0" if s["score"]==3 else "#ffb020" if s["score"]==2 else "#4a6070"
            dir_c = "#00e5a0" if s["direction"]=="long" else "#ff3d55"
            pr_c  = "#00e5a0" if s["prob"]>=65 else "#ffb020" if s["prob"]>=50 else "#ff3d55"
            bdr   = "#00e5a0" if s["direction"]=="long" else "#ff3d55"
            with cols[i % 4]:
                st.markdown(f"""
                <div style="background:#0d1520;border:1px solid #1a2535;
                            border-left:3px solid {bdr};border-radius:10px;
                            padding:13px 15px;margin-bottom:4px">
                  <div style="display:flex;align-items:center;
                              justify-content:space-between;margin-bottom:6px">
                    <span style="font-size:15px;font-weight:700;color:#dde4f0">{s['ticker']}</span>
                    <span style="font-size:10px;font-weight:700;padding:2px 6px;
                                 border-radius:3px;background:rgba(0,0,0,.4);
                                 color:{sc_c}">{s['score']}/3</span>
                  </div>
                  <span style="font-size:11px;font-weight:700;color:{dir_c}">
                    {'▲ LONG' if s['direction']=='long' else '▼ SHORT'}
                  </span>
                  <div style="font-size:19px;font-weight:700;color:#dde4f0;margin:5px 0">
                    ${s['price']:.2f}
                  </div>
                  <div style="font-size:11px;color:#4a6070;display:flex;justify-content:space-between;margin:2px 0">
                    <span>SL</span><span style="color:#ff3d55">${s['sl']:.2f} −{s['sl_pct']:.1f}%</span>
                  </div>
                  <div style="font-size:11px;color:#4a6070;display:flex;justify-content:space-between;margin:2px 0">
                    <span>TP</span><span style="color:#00e5a0">${s['tp']:.2f} +{s['tp_pct']:.1f}%</span>
                  </div>
                  <div style="font-size:11px;color:#4a6070;display:flex;justify-content:space-between;margin:2px 0">
                    <span>Acciones · Cap</span><span style="color:#dde4f0">{s['shares']} · {s['cap_pct']:.1f}%</span>
                  </div>
                  <div style="font-size:11px;color:#4a6070;display:flex;justify-content:space-between;margin:2px 0">
                    <span>Vol · RSI</span><span style="color:#dde4f0">{s['vol']:.1f}x · {s['rsi']:.0f}</span>
                  </div>
                  <div style="font-size:11px;color:#4a6070;display:flex;justify-content:space-between;margin:2px 0">
                    <span>Señales</span><span style="color:#ffb020">{s['signals']}</span>
                  </div>
                  <div style="font-size:11px;color:#4a6070;display:flex;justify-content:space-between;margin:2px 0">
                    <span>Prob ML</span><span style="color:{pr_c};font-weight:700">{s['prob']:.0f}%</span>
                  </div>
                </div>""", unsafe_allow_html=True)

                if st.button("Registrar →", key=f"sig_{s['ticker']}_{i}",
                             use_container_width=True):
                    st.session_state.update({
                        "pf_tk":s["ticker"], "pf_di":s["direction"],
                        "pf_en":s["price"],  "pf_sl":s["sl"],
                        "pf_tp":s["tp"],     "pf_sc":s["score"],
                        "pf_sg":s["signals"],
                    })
                    st.success(f"✅ {s['ticker']} cargado → pestaña Hoy")

# ─── TAB 2 — HOY ────────────────────────────────────────
with tab_hoy:
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("P&L hoy",       f"${pnl_hoy:+.2f}")
    m2.metric("Win rate",      f"{wr_hoy:.0f}%")
    m3.metric("Cerradas",      len(cerr_h))
    m4.metric("Abiertas",      len(abrt_h))
    wins_h  = cerr_h[cerr_h["pnl_usd"]>0]["pnl_usd"].sum()  if len(cerr_h) else 0
    losses_h= abs(cerr_h[cerr_h["pnl_usd"]<0]["pnl_usd"].sum()) if len(cerr_h) else 0
    m5.metric("Profit factor", f"{wins_h/losses_h:.2f}" if losses_h>0 else "—")

    st.divider()

    # Formulario registro
    pf = lambda k,d: st.session_state.get(k, d)
    with st.expander("➕ Nueva operación", expanded=bool(pf("pf_tk",""))):
        ca,cb,cc,cd = st.columns(4)
        tk_in = ca.text_input("Ticker",
                    value=pf("pf_tk",""), placeholder="NVDA").upper()
        di_in = ca.selectbox("Dirección",["long","short"],
                    index=0 if pf("pf_di","long")=="long" else 1)
        sc_in = ca.selectbox("Score",[1,2,3], index=pf("pf_sc",2)-1,
                    format_func=lambda x:f"{x}/3")
        en_in = cb.number_input("Entrada",
                    value=float(pf("pf_en",0.0)), min_value=0.0, step=0.01, format="%.2f")
        sl_in = cb.number_input("Stop Loss",
                    value=float(pf("pf_sl",0.0)), min_value=0.0, step=0.01, format="%.2f")
        tp_in = cc.number_input("Take Profit",
                    value=float(pf("pf_tp",0.0)), min_value=0.0, step=0.01, format="%.2f")
        sg_in = cc.text_input("Señales", value=pf("pf_sg",""))
        no_in = cd.text_input("Notas", placeholder="Opcional")

        if en_in > 0 and sl_in > 0:
            _r  = abs(en_in - sl_in)
            _sh = max(1, int(capital * risk_pct/100 / _r)) if _r > 0 else 1
            _tp = tp_in if tp_in > 0 else (
                en_in + _r*rr_ratio if di_in=="long" else en_in - _r*rr_ratio
            )
            st.caption(
                f"**{_sh} acciones** · Riesgo ${_r*_sh:.2f} · "
                f"SL −{(_r/en_in*100):.2f}% · TP +{(abs(_tp-en_in)/en_in*100):.2f}%"
            )

        if st.button("✅ Registrar trade", type="primary"):
            if not tk_in or en_in<=0 or sl_in<=0:
                st.error("Completa ticker, entrada y SL.")
            else:
                with st.spinner("Guardando en Google Sheets..."):
                    sh = add_trade(
                        capital, risk_pct, rr_ratio,
                        tk_in, di_in, en_in, sl_in,
                        tp_in if tp_in > 0 else None,
                        sc_in, sg_in, no_in,
                    )
                for k in ["pf_tk","pf_di","pf_en","pf_sl","pf_tp","pf_sc","pf_sg"]:
                    st.session_state.pop(k, None)
                st.success(f"✅ {tk_in} {di_in.upper()} guardado en Sheets · {sh} acciones")
                st.rerun()

    # Abiertas
    if not abrt_h.empty:
        st.markdown("**Posiciones abiertas**")
        for _, t in abrt_h.iterrows():
            ca,cb,cc,cd = st.columns([2,2,2,1])
            en_v = float(t["entry"]) if pd.notna(t["entry"]) else 0
            sl_v = float(t["sl"])    if pd.notna(t["sl"])    else 0
            tp_v = float(t["tp1"])   if pd.notna(t["tp1"])   else 0
            ca.markdown(f"**{t['ticker']}** "
                        f"{'▲' if str(t['direction'])=='long' else '▼'} @ ${en_v:.2f}")
            cb.caption(f"SL ${sl_v:.2f} · TP ${tp_v:.2f}")
            ex_v = cc.number_input("Precio salida", key=f"ex_{t['id']}",
                                   min_value=0.0, step=0.01, format="%.2f")
            nota_v = cc.text_input("Nota", key=f"no_{t['id']}", placeholder="Opcional")
            with cd:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                if st.button("Cerrar", key=f"cl_{t['id']}"):
                    if ex_v > 0:
                        with st.spinner("Guardando..."):
                            pnl_c = close_trade(t["id"], ex_v, capital, risk_pct, nota_v)
                        st.success(f"{'🟢' if pnl_c>=0 else '🔴'} ${pnl_c:+.2f}")
                        st.rerun()
                    else:
                        st.error("Precio requerido")
        st.divider()

    st.markdown("**Operaciones del día**")
    if t_hoy.empty:
        st.info("Sin operaciones hoy.")
    else:
        show = ["hora","ticker","direction","entry","exit_price",
                "pnl_usd","score","signals","resultado","notas"]
        df_s = t_hoy[[c for c in show if c in t_hoy.columns]].copy()
        df_s.columns = ["Hora","Ticker","Dir","Entrada","Salida",
                        "P&L $","Score","Señales","Resultado","Notas"][:len(df_s.columns)]
        def hl(row):
            r = str(row.iloc[8]) if len(row)>8 else ""
            if r=="win":  return ["background-color:rgba(0,229,160,.06)"]*len(row)
            if r=="loss": return ["background-color:rgba(255,61,85,.06)"]*len(row)
            return [""]*len(row)
        st.dataframe(df_s.sort_values("Hora",ascending=False).style.apply(hl,axis=1),
                     use_container_width=True, hide_index=True)

# ─── TAB 3 — HISTORIAL ──────────────────────────────────
with tab_his:
    cl_all = trades_num[trades_num["exit_price"].notna()].copy() \
             if not trades_num.empty else pd.DataFrame()

    if cl_all.empty:
        st.info("Sin operaciones cerradas aún. Empieza a registrar desde la pestaña Hoy.")
    else:
        stats = calc_stats(cl_all)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Total ops",     stats.get("n",0))
        m2.metric("Win rate",      f"{stats.get('wr',0):.1f}%")
        m3.metric("P&L acumulado", f"${stats.get('pnl',0):+.2f}")
        m4.metric("Profit factor", stats.get("pf","—"))
        st.divider()

        # Curva equity
        st.markdown("**Curva de equity**")
        sorted_cl = cl_all.sort_values("fecha")
        equity = [capital]
        for p in sorted_cl["pnl_usd"].fillna(0):
            equity.append(equity[-1] + float(p))
        color_eq = "#00e5a0" if equity[-1]>=capital else "#ff3d55"
        rgb = "0,229,160" if color_eq=="#00e5a0" else "255,61,85"
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=equity, mode="lines",
            line=dict(color=color_eq, width=2),
            fill="tozeroy", fillcolor=f"rgba({rgb},.06)",
            hovertemplate="$%{y:.2f}<extra></extra>"))
        fig.add_hline(y=capital, line_dash="dash", line_color="#1a2535", line_width=1)
        fig.update_layout(paper_bgcolor="#06080b", plot_bgcolor="#06080b",
            margin=dict(l=0,r=0,t=4,b=0), height=200, showlegend=False,
            xaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
            yaxis=dict(showgrid=False,zeroline=False,
                       tickfont=dict(color="#4a6070",size=10)))
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

        # Stats en columnas
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**Por ticker**")
            if "by_ticker" in stats:
                for _, r in stats["by_ticker"].iterrows():
                    x1,x2,x3 = st.columns(3)
                    x1.metric(str(r["ticker"]), f"{r['n']:.0f} ops")
                    x2.metric("WR",  f"{r['wr']:.0f}%")
                    x3.metric("P&L", f"${r['pnl']:+.0f}")
        with col_b:
            st.markdown("**Por score**")
            if "by_score" in stats:
                for _, r in stats["by_score"].iterrows():
                    x1,x2,x3 = st.columns(3)
                    x1.metric(f"{int(r['score'])}/3", f"{r['n']:.0f} ops")
                    x2.metric("WR",  f"{r['wr']:.0f}%")
                    x3.metric("P&L", f"${r['pnl']:+.0f}")
        with col_c:
            st.markdown("**Mejores horas**")
            if "by_hora" in stats:
                for _, r in stats["by_hora"].head(5).iterrows():
                    x1,x2,x3 = st.columns(3)
                    x1.metric(f"{r['hh']}:xx", f"{r['n']:.0f} ops")
                    x2.metric("WR",  f"{r['wr']:.0f}%")
                    x3.metric("P&L", f"${r['pnl']:+.0f}")
        st.divider()

        # Tabla filtrable + export
        st.markdown("**Todas las operaciones**")
        f1,f2,f3 = st.columns(3)
        tks_u  = ["Todos"]+sorted(cl_all["ticker"].dropna().unique().tolist())
        fil_tk = f1.selectbox("Ticker",    tks_u)
        fil_di = f2.selectbox("Dirección", ["Todos","long","short"])
        fil_re = f3.selectbox("Resultado", ["Todos","win","loss"])
        df_fil = cl_all.copy()
        if fil_tk!="Todos": df_fil=df_fil[df_fil["ticker"]==fil_tk]
        if fil_di!="Todos": df_fil=df_fil[df_fil["direction"]==fil_di]
        if fil_re!="Todos": df_fil=df_fil[df_fil["resultado"]==fil_re]
        show = ["fecha","hora","ticker","direction","entry","exit_price",
                "pnl_usd","pnl_pct","score","signals","resultado","notas"]
        st.dataframe(
            df_fil[show].rename(columns={
                "fecha":"Fecha","hora":"Hora","ticker":"Ticker","direction":"Dir",
                "entry":"Entrada","exit_price":"Salida","pnl_usd":"P&L $",
                "pnl_pct":"P&L %","score":"Score","signals":"Señales",
                "resultado":"Resultado","notas":"Notas"}),
            use_container_width=True, hide_index=True)
        buf = io.StringIO()
        df_fil.to_csv(buf, index=False)
        st.download_button("⬇ Exportar CSV", data=buf.getvalue(),
            file_name=f"marginedge_{hoy}.csv", mime="text/csv")

# ─── TAB 4 — CALCULADORA ────────────────────────────────
with tab_calc:
    st.markdown("### Calculadora de posición")
    ca, cb = st.columns(2)
    with ca:
        c_en  = st.number_input("Precio entrada", min_value=0.0, step=0.01, format="%.2f", key="cc_en")
        c_sl  = st.number_input("Stop Loss",      min_value=0.0, step=0.01, format="%.2f", key="cc_sl")
        c_cap = st.number_input("Capital ($)",    value=10000.0, step=500.0, key="cc_cap")
    with cb:
        c_ri  = st.slider("Riesgo (%)", 0.5, 3.0, 1.0, 0.1, key="cc_ri")
        c_rr  = st.slider("R:R TP1",   1.0, 5.0, 2.0, 0.5, key="cc_rr")
        c_rr2 = st.slider("R:R TP2",   1.0, 6.0, 3.0, 0.5, key="cc_rr2")
    if c_en > 0 and c_sl > 0:
        c_r   = abs(c_en - c_sl)
        c_sh  = max(1, int(c_cap * c_ri/100 / c_r))
        c_tp1 = c_en + c_r*c_rr
        c_tp2 = c_en + c_r*c_rr2
        c_cu  = c_sh * c_en
        r1,r2 = st.columns(2)
        r1.metric("Acciones",          c_sh)
        r1.metric("Capital inmovilizado", f"${c_cu:,.0f}", f"{c_cu/c_cap*100:.1f}%")
        r1.metric("Riesgo máximo",     f"${c_r*c_sh:.2f}", f"{c_ri}%")
        r2.metric("Stop Loss",         f"${c_sl:.2f}", f"−{c_r/c_en*100:.2f}%")
        r2.metric("Take Profit 1",     f"${c_tp1:.2f}", f"+{c_r*c_rr/c_en*100:.2f}%")
        r2.metric("Take Profit 2",     f"${c_tp2:.2f}", f"+{c_r*c_rr2/c_en*100:.2f}%")
    else:
        st.info("Ingresa precio de entrada y stop loss para calcular.")
