# ╔══════════════════════════════════════════════════════════════╗
# ║  TrackOp — App Streamlit v1.0                               ║
# ║                                                              ║
# ║  Para correr localmente:                                    ║
# ║    pip install -r requirements.txt                          ║
# ║    streamlit run trackop_app.py                             ║
# ║                                                              ║
# ║  Para deploy en Streamlit Cloud:                            ║
# ║    1. Sube este archivo a GitHub                            ║
# ║    2. Conecta el repo en streamlit.io/cloud                 ║
# ║    3. Agrega los secrets de Google Drive en Settings        ║
# ╚══════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import json, os, io, time
from datetime import datetime, timezone, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ════════════════════════════════════════════════════════════════
#  CONFIG DE PÁGINA
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TrackOp",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════
#  CSS — estilo oscuro consistente con el resto del sistema
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#07090c}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #1c2530}
.stMetric{background:#0d1117;border:1px solid #1c2530;border-radius:8px;padding:12px}
.stMetric label{color:#6a7f96 !important;font-size:11px}
.stDataFrame{border:1px solid #1c2530;border-radius:8px}
div[data-testid="metric-container"]{background:#0d1117;border:1px solid #1c2530;border-radius:8px;padding:12px}
.stButton>button{background:#131a22;border:1px solid #263040;color:#dde4f0;border-radius:6px;font-size:12px}
.stButton>button:hover{background:#192030;border-color:#3db8ff;color:#3db8ff}
h1,h2,h3{color:#dde4f0 !important}
.stSelectbox label,.stNumberInput label,.stTextInput label{color:#6a7f96 !important;font-size:11px}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════
CAPITAL        = 10_000.0
RISK_PCT       = 0.015
RR_RATIO       = 2.0
ATR_MULT       = 0.5
INTERVAL       = "15m"
PERIOD         = "5d"
WIN_RATE       = 0.48
DAILY_LOSS_LIM = 0.03

MERCADOS = {
    "🇺🇸 US Large Cap": [
        "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA",
        "JPM","V","WMT","MA","HD","XOM","MRK","UNH",
    ],
    "🇺🇸 US Tech & Growth": [
        "AMD","INTC","ADBE","CRM","NOW","CRWD","NET",
        "NFLX","UBER","COIN","PYPL","PLTR","ARM","SHOP",
    ],
    "🇲🇽 BMV México": [
        "AMXL.MX","WALMEX.MX","GFNORTEO.MX","CEMEXCPO.MX","BIMBOA.MX",
    ],
}

# ════════════════════════════════════════════════════════════════
#  HORA CDMX
# ════════════════════════════════════════════════════════════════
def cdmx_now():
    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=6)

def us_dst():
    n = cdmx_now(); y = n.year
    m = datetime(y,3,1);  s = m + timedelta(days=(6-m.weekday()+7)%7+7)
    v = datetime(y,11,1); e = v + timedelta(days=(6-v.weekday())%7)
    return s <= n < e

DST = us_dst()
MKT_OPEN  = 7  if DST else 8
MKT_CLOSE = 14 if DST else 15
open_str  = "7:30 AM" if DST else "8:30 AM"
close_str = "2:00 PM" if DST else "3:00 PM"
h_now = cdmx_now().hour + cdmx_now().minute/60
mercado_abierto = (MKT_OPEN + 0.5) <= h_now < MKT_CLOSE

# ════════════════════════════════════════════════════════════════
#  GOOGLE DRIVE — almacenamiento persistente
#  En Streamlit Cloud usa secrets. Localmente usa archivo directo.
# ════════════════════════════════════════════════════════════════
TRADES_FILE = "trades_reales.csv"

@st.cache_resource
def get_drive_client():
    """Conecta con Google Drive vía service account (solo en Cloud)."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds_dict = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT"])
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/drive",
                    "https://www.googleapis.com/auth/spreadsheets"]
        )
        return gspread.authorize(creds)
    except:
        return None  # fallback a archivo local

def cargar_trades():
    """Carga el historial de trades."""
    cols = ["id","fecha","ticker","direction","entry","sl","tp1",
            "exit_price","resultado","pnl_usd","pnl_pct",
            "score","prob_ml","rsi","vol_ratio","signals",
            "hora_entrada","notas"]
    if os.path.exists(TRADES_FILE):
        try:
            df = pd.read_csv(TRADES_FILE)
            # asegurar que tiene todas las columnas
            for c in cols:
                if c not in df.columns: df[c] = None
            return df
        except:
            pass
    return pd.DataFrame(columns=cols)

def guardar_trades(df):
    """Guarda el historial."""
    df.to_csv(TRADES_FILE, index=False)

def agregar_trade(ticker, direction, entry, sl, tp1,
                  score=1, prob_ml=50, rsi=50, vol_ratio=1,
                  signals="", hora_entrada="", notas=""):
    df = cargar_trades()
    riesgo = abs(entry - sl) if sl else entry * RISK_PCT
    shares = max(1, int(CAPITAL * RISK_PCT / riesgo)) if riesgo > 0 else 1
    nueva = {
        "id"          : str(int(time.time())),
        "fecha"       : cdmx_now().strftime("%Y-%m-%d"),
        "ticker"      : ticker.upper(),
        "direction"   : direction,
        "entry"       : entry,
        "sl"          : sl,
        "tp1"         : tp1,
        "exit_price"  : None,
        "resultado"   : None,
        "pnl_usd"     : None,
        "pnl_pct"     : None,
        "score"       : score,
        "prob_ml"     : prob_ml,
        "rsi"         : rsi,
        "vol_ratio"   : vol_ratio,
        "signals"     : signals,
        "hora_entrada": hora_entrada or cdmx_now().strftime("%H:%M"),
        "notas"       : notas,
    }
    df = pd.concat([df, pd.DataFrame([nueva])], ignore_index=True)
    guardar_trades(df)
    return df

def cerrar_trade_id(trade_id, exit_price, notas=""):
    df = cargar_trades()
    mask = df["id"].astype(str) == str(trade_id)
    if not mask.any(): return df, False
    idx   = df[mask].index[0]
    entry = float(df.loc[idx, "entry"])
    dirn  = str(df.loc[idx, "direction"])
    sl    = float(df.loc[idx, "sl"])
    shares= max(1, int(CAPITAL*RISK_PCT/max(abs(entry-sl),0.01)))
    diff  = (exit_price-entry) if dirn=="long" else (entry-exit_price)
    pnl   = round(diff*shares, 2)
    df.loc[idx, "exit_price"] = exit_price
    df.loc[idx, "pnl_usd"]   = pnl
    df.loc[idx, "pnl_pct"]   = round(diff/entry*100, 3)
    df.loc[idx, "resultado"]  = "win" if pnl > 0 else "loss"
    df.loc[idx, "notas"]      = notas or df.loc[idx,"notas"]
    guardar_trades(df)
    return df, pnl

# ════════════════════════════════════════════════════════════════
#  INDICADORES
# ════════════════════════════════════════════════════════════════
def ema(s,p): return s.ewm(span=p,adjust=False).mean()
def sma(s,p): return s.rolling(p).mean()
def xover(a,b): return (a>b)&(a.shift()<=b.shift())
def xunder(a,b): return (a<b)&(a.shift()>=b.shift())

def rsi_calc(s, p=14):
    d=s.diff(); g=d.clip(lower=0).ewm(alpha=1/p,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(alpha=1/p,adjust=False).mean()
    return 100-(100/(1+g/l.replace(0,np.nan)))

def atr_calc(h,l,c,p=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/p,adjust=False).mean()

def vwap_calc(h,l,c,vol):
    hlc3=(h+l+c)/3
    return (hlc3*vol).groupby(hlc3.index.date).cumsum()/\
           vol.groupby(vol.index.date).cumsum()

def compute(df):
    d=df.copy()
    d["e9"]  =ema(d["Close"],9);  d["e21"]=ema(d["Close"],21)
    d["e50"] =ema(d["Close"],50); d["vwap"]=vwap_calc(d["High"],d["Low"],d["Close"],d["Volume"])
    d["rsi"] =rsi_calc(d["Close"]); d["atr"]=atr_calc(d["High"],d["Low"],d["Close"])
    d["vavg"]=sma(d["Volume"],20)
    v1=(d["Volume"]>d["vavg"]*1.5); v2=(d["Volume"]>d["vavg"]*2.0)
    al=(d["Close"]>d["e50"])&(d["e21"]>d["e50"])
    bj=(d["Close"]<d["e50"])&(d["e21"]<d["e50"])
    rl=d["rsi"]<70; rs=d["rsi"]>30
    s1l=xover(d["Close"],d["vwap"])&v1&al&rl
    s1s=xunder(d["Close"],d["vwap"])&v1&bj&rs
    tl=(d["Low"]<=d["e21"]*1.003)&(d["Close"]>d["e21"])
    ts=(d["High"]>=d["e21"]*0.997)&(d["Close"]<d["e21"])
    s2l=tl&~tl.shift(fill_value=False)&al&rl
    s2s=ts&~ts.shift(fill_value=False)&bj&rs
    ph=d["High"].shift().rolling(20).max(); pl=d["Low"].shift().rolling(20).min()
    s3l=xover(d["Close"],ph)&v2&(d["Close"]>d["e21"])&rl
    s3s=xunder(d["Close"],pl)&v2&(d["Close"]<d["e21"])&rs
    d["scl"]=s1l.astype(int)+s2l.astype(int)+s3l.astype(int)
    d["scs"]=s1s.astype(int)+s2s.astype(int)+s3s.astype(int)
    return d

# ════════════════════════════════════════════════════════════════
#  ML
# ════════════════════════════════════════════════════════════════
def get_prob(df_price, direction, trades_df, ticker):
    sc_col = "scl" if direction=="long" else "scs"
    sig    = df_price["scl"]>=2 if direction=="long" else df_price["scs"]>=2
    rows   = []
    for i in range(40, len(df_price)-16):
        if not sig.iloc[i]: continue
        r = df_price.iloc[i]
        if pd.isna(r["rsi"]) or pd.isna(r["atr"]): continue
        e = df_price["Open"].iloc[i+1]
        if e<=0 or pd.isna(e): continue
        sl = (df_price["Low"].iloc[max(0,i-2):i+1].min()-r["atr"]*ATR_MULT) \
             if direction=="long" \
             else (df_price["High"].iloc[max(0,i-2):i+1].max()+r["atr"]*ATR_MULT)
        tp = e+(e-sl)*RR_RATIO if direction=="long" else e-(sl-e)*RR_RATIO
        feat = [float(r["rsi"]),r["atr"]/e if e>0 else 0,
                (r["Close"]-r["vwap"])/r["vwap"]*100 if r["vwap"]>0 else 0,
                (r["Close"]-r["e21"])/r["e21"]*100,
                (r["Close"]-r["e50"])/r["e50"]*100,
                min(r["Volume"]/r["vavg"] if r["vavg"]>0 else 1,10),
                int(df_price[sc_col].iloc[i]),
                (r["Close"]/df_price["Close"].iloc[i-1]-1)*100 if i>1 else 0]
        lbl=0
        for j in range(i+1,min(i+16,len(df_price))):
            b=df_price.iloc[j]
            if direction=="long":
                if b["Low"]<=sl:  lbl=0;break
                if b["High"]>=tp: lbl=1;break
            else:
                if b["High"]>=sl: lbl=0;break
                if b["Low"]<=tp:  lbl=1;break
        rows.append((feat,lbl,1.0))
    # Trades reales — 3× peso
    if not trades_df.empty:
        rt = trades_df[(trades_df["ticker"]==ticker.upper()) &
                       (trades_df["exit_price"].notna()) &
                       (trades_df["direction"]==direction)]
        for _,t in rt.iterrows():
            try:
                feat_r=[float(t.get("rsi",50)),0.01,0,0,0,
                        float(t.get("vol_ratio",1.5)),
                        float(t.get("score",2)),0]
                rows.append((feat_r,1 if t.get("resultado","loss")=="win" else 0, 3.0))
            except: pass
    if len(rows)<12: return 50.0
    X=np.array([r[0] for r in rows],dtype=float)
    y=np.array([r[1] for r in rows],dtype=int)
    w=np.array([r[2] for r in rows],dtype=float)
    mask=np.isfinite(X).all(axis=1)
    X,y,w=X[mask],y[mask],w[mask]
    if len(X)<12 or len(np.unique(y))<2: return 50.0
    sc=StandardScaler(); Xs=sc.fit_transform(X)
    m=RandomForestClassifier(n_estimators=80,max_depth=5,
      min_samples_leaf=max(2,len(X)//20),
      class_weight="balanced",random_state=42,n_jobs=-1)
    m.fit(Xs,y,sample_weight=w)
    fn=[float(df_price["rsi"].iloc[-1]),
        df_price["atr"].iloc[-1]/df_price["Close"].iloc[-1],
        (df_price["Close"].iloc[-1]-df_price["vwap"].iloc[-1])/
        df_price["vwap"].iloc[-1]*100 if df_price["vwap"].iloc[-1]>0 else 0,
        (df_price["Close"].iloc[-1]-df_price["e21"].iloc[-1])/df_price["e21"].iloc[-1]*100,
        (df_price["Close"].iloc[-1]-df_price["e50"].iloc[-1])/df_price["e50"].iloc[-1]*100,
        min(df_price["Volume"].iloc[-1]/df_price["vavg"].iloc[-1]
            if df_price["vavg"].iloc[-1]>0 else 1,10),
        int(df_price[sc_col].iloc[-1]),
        (df_price["Close"].iloc[-1]/df_price["Close"].iloc[-2]-1)*100 if len(df_price)>1 else 0]
    if not all(np.isfinite(fn)): return 50.0
    return round(float(m.predict_proba(sc.transform([fn]))[0][1])*100,1)

# ════════════════════════════════════════════════════════════════
#  SCANNER
# ════════════════════════════════════════════════════════════════
@st.cache_data(ttl=900)   # cache 15 minutos
def scan_ticker(ticker, _trades_hash):
    """Analiza un ticker. Cache de 15 min para no re-descargar."""
    try:
        raw=yf.download(ticker,period=PERIOD,interval=INTERVAL,
                        auto_adjust=True,progress=False)
        if isinstance(raw.columns,pd.MultiIndex):
            raw.columns=raw.columns.get_level_values(0)
        if raw.empty or len(raw)<60: return None
    except: return None
    df=compute(raw)
    for offset in range(5):
        idx=-(1+offset)
        try: row=df.iloc[idx]
        except: continue
        if pd.isna(row["rsi"]) or pd.isna(row["atr"]): continue
        scl=int(row["scl"]); scs=int(row["scs"])
        if max(scl,scs)<1: continue
        dirn="long" if scl>=scs else "short"
        score=max(scl,scs)
        vr=row["Volume"]/row["vavg"] if row["vavg"]>0 else 1
        thr=1.0*(0.6 if offset==0 else 1.0)
        if vr<thr: continue
        if dirn=="long"  and row["rsi"]>72: continue
        if dirn=="short" and row["rsi"]<28: continue
        price=float(row["Close"])
        sl=(price-row["atr"]*ATR_MULT*2) if dirn=="long" \
           else (price+row["atr"]*ATR_MULT*2)
        tp=price+(price-sl)*RR_RATIO if dirn=="long" \
           else price-(sl-price)*RR_RATIO
        rps=abs(price-sl)
        shares=max(1,int(CAPITAL*RISK_PCT/rps)) if rps>0 else 1
        active=[]
        pv=df["Close"].iloc[idx-1] if abs(idx)>1 else price
        if dirn=="long":
            if price>row["vwap"] and pv<=row["vwap"]: active.append("VWAP")
            if row["Low"]<=row["e21"]*1.003: active.append("EMA")
            ph=df["High"].shift().rolling(20).max().iloc[idx]
            if price>ph: active.append("Break")
        else:
            if price<row["vwap"] and pv>=row["vwap"]: active.append("VWAP")
            if row["High"]>=row["e21"]*0.997: active.append("EMA")
            pl=df["Low"].shift().rolling(20).min().iloc[idx]
            if price<pl: active.append("Break")
        return {
            "ticker": ticker,"direction": dirn,"score": score,
            "price": round(price,2),"sl": round(sl,2),"tp": round(tp,2),
            "sl_pct": round(abs(sl-price)/price*100,2),
            "tp_pct": round(abs(tp-price)/price*100,2),
            "shares": shares,
            "cap_pct": round(shares*price/CAPITAL*100,1),
            "vol": round(vr,1),"rsi": round(float(row["rsi"]),1),
            "signals": "+".join(active) if active else f"score{score}",
            "hora": str(df.index[idx])[11:16],
            "df_ref": df,
        }
    return None

def run_scan(trades_df, tickers_sel=None):
    all_tickers = tickers_sel or [t for v in MERCADOS.values() for t in v]
    trades_hash = len(trades_df) if not trades_df.empty else 0
    resultados  = []
    prog = st.progress(0, text="Escaneando tickers...")
    for i, tk in enumerate(all_tickers):
        r = scan_ticker(tk, trades_hash)
        if r:
            try:
                prob = get_prob(r["df_ref"], r["direction"], trades_df, tk)
            except:
                prob = 50.0
            r["prob"] = prob
            r.pop("df_ref", None)
            resultados.append(r)
        prog.progress((i+1)/len(all_tickers),
                      text=f"Escaneando... {tk} ({i+1}/{len(all_tickers)})")
    prog.empty()
    return sorted(resultados, key=lambda x:(x["score"],x["prob"]), reverse=True)

# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## TrackOp 📈")

    # Estado del mercado
    if mercado_abierto:
        st.success(f"🟢 MERCADO ABIERTO\n{open_str} – {close_str} CDMX")
    else:
        st.warning(f"🟡 MERCADO CERRADO\nAbre {open_str} CDMX")

    st.caption(f"{'🕐 DST USA activo' if DST else '🕐 Hora estándar USA'}")
    st.caption(f"Hora CDMX: {cdmx_now().strftime('%H:%M:%S')}")

    st.divider()

    # Config rápida
    st.markdown("### Config")
    capital  = st.number_input("Capital ($)", value=CAPITAL, step=500.0, format="%.0f")
    risk_pct = st.slider("Riesgo/op (%)", 0.5, 3.0, 1.5, 0.1)
    min_prob = st.slider("Prob ML mínima (%)", 0, 100, 50, 5)
    min_score= st.selectbox("Score mínimo", [1,2,3], index=0,
                             format_func=lambda x:f"{x}/3")

    st.divider()

    # Filtro de mercados
    st.markdown("### Mercados")
    mkt_sel = {}
    for m in MERCADOS:
        mkt_sel[m] = st.checkbox(m, value=True)

    st.divider()
    if st.button("🔄 Actualizar señales", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ════════════════════════════════════════════════════════════════
#  TABS PRINCIPALES
# ════════════════════════════════════════════════════════════════
tab_senales, tab_trades, tab_historial, tab_cuenta, tab_calc = st.tabs([
    "📊 Señales", "➕ Registrar Trade", "📋 Historial", "💰 Cuenta", "🔢 Calculadora"
])

# ════════════════════════════════════════════════════════════════
#  TAB 1 — SEÑALES
# ════════════════════════════════════════════════════════════════
with tab_senales:

    trades_df = cargar_trades()
    tickers_filtrados = [t for m,tks in MERCADOS.items()
                         if mkt_sel.get(m,True) for t in tks]

    with st.spinner("Cargando señales..."):
        senales = run_scan(trades_df, tickers_filtrados)

    # Filtrar según config del sidebar
    senales_filtradas = [s for s in senales
                         if s["prob"] >= min_prob and s["score"] >= min_score]

    # Métricas rápidas
    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("Señales activas", len(senales_filtradas))
    col2.metric("NYSE", "ABIERTO" if mercado_abierto else "CERRADO",
                delta=f"{open_str}–{close_str}")
    longs  = sum(1 for s in senales_filtradas if s["direction"]=="long")
    shorts = sum(1 for s in senales_filtradas if s["direction"]=="short")
    col3.metric("LONGs / SHORTs", f"{longs} / {shorts}")
    top    = next((s for s in senales_filtradas if s["score"]==3),None)
    col4.metric("Mayor score", f"{top['ticker']} {top['score']}/3" if top else "—")
    col5.metric("Prob promedio",
                f"{np.mean([s['prob'] for s in senales_filtradas]):.0f}%" if senales_filtradas else "—")

    st.divider()

    if not senales_filtradas:
        st.info("Sin señales que cumplan los filtros. Baja el score mínimo o la prob mínima en la barra lateral.")
    else:
        # Tabla de señales — una por mercado
        for mercado, tickers in MERCADOS.items():
            if not mkt_sel.get(mercado, True): continue
            rows = [s for s in senales_filtradas if s["ticker"] in tickers]
            if not rows: continue

            st.markdown(f"### {mercado}")
            df_show = pd.DataFrame(rows)

            # Formatear para mostrar
            df_table = pd.DataFrame({
                "Ticker"    : df_show["ticker"],
                "Acción"    : df_show.apply(lambda r:
                    "▲ LONG" if r["direction"]=="long" else "▼ SHORT", axis=1),
                "Score"     : df_show["score"].apply(lambda x:f"{x}/3"),
                "Señales"   : df_show["signals"],
                "Precio"    : df_show["price"].apply(lambda x:f"${x:.2f}"),
                "Stop Loss" : df_show.apply(lambda r:
                    f"${r['sl']:.2f} (−{r['sl_pct']:.1f}%)", axis=1),
                "Take Profit":df_show.apply(lambda r:
                    f"${r['tp']:.2f} (+{r['tp_pct']:.1f}%)", axis=1),
                "Acciones"  : df_show["shares"],
                "% Capital" : df_show["cap_pct"].apply(lambda x:f"{x:.1f}%"),
                "Vol"       : df_show["vol"].apply(lambda x:f"{x:.1f}x"),
                "RSI"       : df_show["rsi"].apply(lambda x:f"{x:.0f}"),
                "Prob ML"   : df_show["prob"].apply(lambda x:f"{x:.0f}%"),
                "Hora"      : df_show["hora"],
            })

            # Colorear filas
            def color_row(row):
                base = []
                for col in row.index:
                    if col == "Acción":
                        if "LONG" in str(row[col]):
                            base.append("color: #00e08a; font-weight: bold")
                        else:
                            base.append("color: #ff3d55; font-weight: bold")
                    elif col == "Prob ML":
                        prob_val = float(str(row[col]).replace("%",""))
                        if prob_val >= 65:
                            base.append("color: #00e08a; font-weight: bold")
                        elif prob_val >= 50:
                            base.append("color: #ffb830")
                        else:
                            base.append("color: #ff3d55")
                    elif col == "Score":
                        if row[col] == "3/3":
                            base.append("color: #00e08a; font-weight: bold")
                        elif row[col] == "2/3":
                            base.append("color: #ffb830; font-weight: bold")
                        else:
                            base.append("color: #6a7f96")
                    else:
                        base.append("")
                return base

            styled = df_table.style.apply(color_row, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Botones de registro rápido para cada señal
            st.markdown("**Registrar señal en un click:**")
            cols = st.columns(min(len(rows), 4))
            for i, r in enumerate(rows[:4]):
                with cols[i]:
                    sym = "▲" if r["direction"]=="long" else "▼"
                    btn_label = f"{sym} {r['ticker']}\n${r['price']:.2f} · {r['prob']:.0f}%"
                    if st.button(btn_label, key=f"btn_{r['ticker']}_{i}",
                                 use_container_width=True):
                        # Pre-rellenar el formulario y redirigir al tab de registro
                        st.session_state["pre_ticker"]    = r["ticker"]
                        st.session_state["pre_direction"] = r["direction"]
                        st.session_state["pre_entry"]     = r["price"]
                        st.session_state["pre_sl"]        = r["sl"]
                        st.session_state["pre_tp"]        = r["tp"]
                        st.session_state["pre_score"]     = r["score"]
                        st.session_state["pre_prob"]      = r["prob"]
                        st.session_state["pre_rsi"]       = r["rsi"]
                        st.session_state["pre_vol"]       = r["vol"]
                        st.session_state["pre_signals"]   = r["signals"]
                        st.success(f"✅ {r['ticker']} cargado — ve a la pestaña 'Registrar Trade'")

# ════════════════════════════════════════════════════════════════
#  TAB 2 — REGISTRAR TRADE
# ════════════════════════════════════════════════════════════════
with tab_trades:
    st.markdown("### Registrar nueva operación")
    st.caption("Los campos se pre-rellenan automáticamente al hacer click en una señal de la tabla.")

    # Pre-rellenar desde sesión si viene de la tabla
    pre = lambda k, d: st.session_state.get(k, d)

    c1, c2, c3 = st.columns(3)
    with c1:
        ticker_in = st.text_input("Ticker", value=pre("pre_ticker",""),
                                  placeholder="NVDA").upper()
        direction_in = st.selectbox("Dirección",
            ["long","short"],
            index=0 if pre("pre_direction","long")=="long" else 1)
        score_in = st.selectbox("Score", [1,2,3],
            index=pre("pre_score",2)-1,
            format_func=lambda x:f"{x}/3 — {'débil' if x==1 else 'confirmada' if x==2 else 'máxima'}")

    with c2:
        entry_in = st.number_input("Precio de entrada", value=float(pre("pre_entry",0.0)),
                                    min_value=0.0, step=0.01, format="%.2f")
        sl_in    = st.number_input("Stop Loss",    value=float(pre("pre_sl",0.0)),
                                    min_value=0.0, step=0.01, format="%.2f")
        tp_in    = st.number_input("Take Profit 1",value=float(pre("pre_tp",0.0)),
                                    min_value=0.0, step=0.01, format="%.2f")

    with c3:
        prob_in    = st.number_input("Prob ML (%)", value=float(pre("pre_prob",50.0)),
                                      min_value=0.0, max_value=100.0, step=1.0)
        rsi_in     = st.number_input("RSI",         value=float(pre("pre_rsi",50.0)),
                                      min_value=0.0, max_value=100.0, step=1.0)
        vol_in     = st.number_input("Vol ratio",   value=float(pre("pre_vol",1.5)),
                                      min_value=0.0, step=0.1)
        signals_in = st.text_input("Señales", value=pre("pre_signals",""),
                                   placeholder="VWAP+EMA")
        notas_in   = st.text_input("Notas", placeholder="Opcional")

    # Preview del riesgo calculado
    if entry_in > 0 and sl_in > 0:
        riesgo_usd = abs(entry_in - sl_in)
        shares_calc = max(1, int(CAPITAL * risk_pct/100 / riesgo_usd))
        cap_uso = shares_calc * entry_in
        st.info(
            f"**Resumen:** {shares_calc} acciones  ·  "
            f"Capital: ${cap_uso:,.0f} ({cap_uso/CAPITAL*100:.1f}%)  ·  "
            f"Riesgo: ${riesgo_usd*shares_calc:.2f}  ·  "
            f"SL: −{(riesgo_usd/entry_in*100):.2f}%  ·  "
            f"TP1: +{(abs(tp_in-entry_in)/entry_in*100):.2f}%"
        )

    col_btn1, col_btn2 = st.columns([1,3])
    with col_btn1:
        if st.button("✅ Registrar trade", type="primary", use_container_width=True):
            if not ticker_in or entry_in <= 0 or sl_in <= 0 or tp_in <= 0:
                st.error("Completa ticker, entrada, SL y TP1.")
            else:
                agregar_trade(
                    ticker=ticker_in, direction=direction_in,
                    entry=entry_in, sl=sl_in, tp1=tp_in,
                    score=score_in, prob_ml=prob_in,
                    rsi=rsi_in, vol_ratio=vol_in,
                    signals=signals_in, notas=notas_in,
                )
                # Limpiar pre-fill
                for k in ["pre_ticker","pre_direction","pre_entry","pre_sl",
                          "pre_tp","pre_score","pre_prob","pre_rsi",
                          "pre_vol","pre_signals"]:
                    st.session_state.pop(k, None)
                st.success(f"✅ {ticker_in} {direction_in.upper()} registrado en Drive")
                st.rerun()

    st.divider()
    st.markdown("### Cerrar trade abierto")

    trades_df = cargar_trades()
    abiertos  = trades_df[trades_df["exit_price"].isna()] \
                if not trades_df.empty else pd.DataFrame()

    if abiertos.empty:
        st.info("Sin posiciones abiertas.")
    else:
        for _, t in abiertos.iterrows():
            with st.container():
                ca,cb,cc = st.columns([2,2,1])
                with ca:
                    st.markdown(f"**{t['ticker']}** {t['direction'].upper()} "
                                f"@ ${float(t['entry']):.2f}  "
                                f"SL: ${float(t['sl']):.2f}  "
                                f"TP1: ${float(t['tp1']):.2f}")
                with cb:
                    exit_key = f"exit_{t['id']}"
                    nota_key = f"nota_{t['id']}"
                    exit_px  = st.number_input("Precio salida",
                        key=exit_key, min_value=0.0, step=0.01, format="%.2f")
                    nota_c   = st.text_input("Nota", key=nota_key,
                                             placeholder="Opcional")
                with cc:
                    st.markdown("&nbsp;", unsafe_allow_html=True)
                    if st.button("Cerrar", key=f"close_{t['id']}",
                                 type="primary"):
                        if exit_px > 0:
                            _, pnl = cerrar_trade_id(t["id"], exit_px, nota_c)
                            color = "🟢" if pnl >= 0 else "🔴"
                            st.success(f"{color} {t['ticker']} cerrado  P&L: ${pnl:+.2f}")
                            st.rerun()
                        else:
                            st.error("Ingresa el precio de salida")
                st.divider()

# ════════════════════════════════════════════════════════════════
#  TAB 3 — HISTORIAL
# ════════════════════════════════════════════════════════════════
with tab_historial:
    trades_df = cargar_trades()
    cerrados  = trades_df[trades_df["exit_price"].notna()] \
                if not trades_df.empty else pd.DataFrame()

    st.markdown("### Historial de operaciones")

    if cerrados.empty:
        st.info("Sin operaciones cerradas aún.")
    else:
        # Filtros
        f1,f2,f3,f4 = st.columns(4)
        with f1:
            tks_uniq = ["Todos"] + sorted(cerrados["ticker"].unique().tolist())
            fil_tk = st.selectbox("Ticker", tks_uniq)
        with f2:
            fil_dir = st.selectbox("Dirección", ["Todos","long","short"])
        with f3:
            fil_res = st.selectbox("Resultado", ["Todos","win","loss"])
        with f4:
            fil_score = st.selectbox("Score mínimo", [1,2,3],
                                     format_func=lambda x:f"≥ {x}/3")

        df_fil = cerrados.copy()
        if fil_tk  != "Todos": df_fil=df_fil[df_fil["ticker"]==fil_tk]
        if fil_dir != "Todos": df_fil=df_fil[df_fil["direction"]==fil_dir]
        if fil_res != "Todos": df_fil=df_fil[df_fil["resultado"]==fil_res]
        df_fil = df_fil[df_fil["score"].fillna(1).astype(float)>=fil_score]

        # Métricas del filtro
        m1,m2,m3,m4 = st.columns(4)
        pnl_t = df_fil["pnl_usd"].sum() if len(df_fil) else 0
        wr    = (df_fil["resultado"]=="win").mean()*100 if len(df_fil) else 0
        m1.metric("Operaciones", len(df_fil))
        m2.metric("P&L total", f"${pnl_t:+,.2f}")
        m3.metric("Win rate", f"{wr:.1f}%")
        wins_sum  = df_fil[df_fil["pnl_usd"]>0]["pnl_usd"].sum()
        losses_sum= abs(df_fil[df_fil["pnl_usd"]<0]["pnl_usd"].sum())
        pf = f"{wins_sum/losses_sum:.2f}" if losses_sum > 0 else "∞"
        m4.metric("Profit factor", pf)

        # Tabla
        df_show = df_fil[["fecha","ticker","direction","entry","exit_price",
                           "sl","tp1","pnl_usd","pnl_pct","score",
                           "prob_ml","signals","resultado","notas"]].copy()
        df_show.columns = ["Fecha","Ticker","Dir","Entrada","Salida",
                            "SL","TP1","P&L $","P&L %","Score",
                            "Prob ML","Señales","Resultado","Notas"]

        def highlight_res(row):
            if str(row.get("Resultado","")) == "win":
                return ["background-color: rgba(0,224,138,0.08)"]*len(row)
            elif str(row.get("Resultado","")) == "loss":
                return ["background-color: rgba(255,61,85,0.08)"]*len(row)
            return [""]*len(row)

        st.dataframe(
            df_show.style.apply(highlight_res, axis=1),
            use_container_width=True, hide_index=True
        )

        # Export
        csv_buf = io.StringIO()
        df_fil.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇ Exportar CSV",
            data=csv_buf.getvalue(),
            file_name=f"trackop_trades_{cdmx_now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# ════════════════════════════════════════════════════════════════
#  TAB 4 — CUENTA
# ════════════════════════════════════════════════════════════════
with tab_cuenta:
    trades_df = cargar_trades()
    cerrados  = trades_df[trades_df["exit_price"].notna()] \
                if not trades_df.empty else pd.DataFrame()
    abiertos  = trades_df[trades_df["exit_price"].isna()] \
                if not trades_df.empty else pd.DataFrame()

    st.markdown("### Estado de cuenta")
    hoy = cdmx_now().strftime("%Y-%m-%d")

    pnl_total = cerrados["pnl_usd"].sum() if len(cerrados) else 0
    pnl_hoy   = cerrados[cerrados["fecha"]==hoy]["pnl_usd"].sum() \
                if len(cerrados) else 0
    wr_total  = (cerrados["resultado"]=="win").mean()*100 if len(cerrados) else 0
    cap_actual= CAPITAL + pnl_total

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Capital actual",  f"${cap_actual:,.2f}", f"{pnl_total:+.2f}")
    m2.metric("P&L hoy",         f"${pnl_hoy:+.2f}",
              delta=f"{len(cerrados[cerrados['fecha']==hoy])} trades hoy" if len(cerrados) else "0 trades")
    m3.metric("P&L total",       f"${pnl_total:+.2f}")
    m4.metric("Win rate",        f"{wr_total:.1f}%",
              f"{len(cerrados[cerrados['resultado']=='win'])} W / {len(cerrados[cerrados['resultado']=='loss'])} L" if len(cerrados) else "—")
    m5.metric("Trades cerrados", len(cerrados),
              f"{len(abiertos)} abiertos")

    st.divider()

    # Curva de equity simple
    if len(cerrados) > 0:
        st.markdown("### Curva de equity")
        sorted_c = cerrados.sort_values("fecha")
        equity   = [CAPITAL] + [CAPITAL + sorted_c["pnl_usd"].iloc[:i+1].sum()
                                 for i in range(len(sorted_c))]
        eq_df    = pd.DataFrame({
            "Trade": range(len(equity)),
            "Equity": equity
        })
        st.line_chart(eq_df.set_index("Trade"), color="#00e08a")

    # Trades de hoy
    hoy_df = cerrados[cerrados["fecha"]==hoy] if len(cerrados) else pd.DataFrame()
    if not hoy_df.empty:
        st.markdown("### Operaciones de hoy")
        st.dataframe(
            hoy_df[["ticker","direction","entry","exit_price","pnl_usd","resultado","signals","notas"]],
            use_container_width=True, hide_index=True
        )

# ════════════════════════════════════════════════════════════════
#  TAB 5 — CALCULADORA
# ════════════════════════════════════════════════════════════════
with tab_calc:
    st.markdown("### Calculadora de posición")

    cc1, cc2 = st.columns(2)
    with cc1:
        c_entry = st.number_input("Precio de entrada", min_value=0.0,
                                   step=0.01, format="%.2f", key="c_entry")
        c_sl    = st.number_input("Stop Loss",    min_value=0.0,
                                   step=0.01, format="%.2f", key="c_sl")
        c_cap   = st.number_input("Capital ($)", value=10000.0, step=500.0, key="c_cap")
    with cc2:
        c_risk  = st.slider("Riesgo (%)", 0.5, 3.0, 1.5, 0.1, key="c_risk")
        c_rr    = st.slider("R:R TP1", 1.0, 5.0, 2.0, 0.5, key="c_rr")
        c_rr2   = st.slider("R:R TP2", 1.0, 6.0, 3.0, 0.5, key="c_rr2")

    if c_entry > 0 and c_sl > 0:
        riesgo_usd = abs(c_entry - c_sl)
        shares_c   = max(1, int(c_cap * c_risk/100 / riesgo_usd))
        tp1_c      = c_entry + riesgo_usd * c_rr
        tp2_c      = c_entry + riesgo_usd * c_rr2
        cap_uso_c  = shares_c * c_entry

        r1,r2 = st.columns(2)
        with r1:
            st.markdown("#### Resultado")
            st.metric("Acciones a comprar",  shares_c)
            st.metric("Capital inmovilizado", f"${cap_uso_c:,.0f}",
                      f"{cap_uso_c/c_cap*100:.1f}% del capital")
            st.metric("Riesgo máximo",        f"${riesgo_usd*shares_c:.2f}",
                      f"{c_risk}% del capital")
        with r2:
            st.markdown("#### Niveles")
            st.metric("Stop Loss",    f"${c_sl:.2f}",
                      f"−{riesgo_usd/c_entry*100:.2f}%")
            st.metric("Take Profit 1",f"${tp1_c:.2f}",
                      f"+{riesgo_usd*c_rr/c_entry*100:.2f}%")
            st.metric("Take Profit 2",f"${tp2_c:.2f}",
                      f"+{riesgo_usd*c_rr2/c_entry*100:.2f}%")
    else:
        st.info("Ingresa precio de entrada y stop loss para calcular.")
