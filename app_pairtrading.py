"""
OMNI-PAIR SYSTEM v2026.1 - Web Edition
App Streamlit per Pair Trading
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from datetime import datetime, timedelta
import requests
import warnings
import os
import json

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAZIONE
# ============================================================
st.set_page_config(
    page_title="OMNI-PAIR System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FILE = "omni_pair.db"
CONFIG_FILE = "omni_config.json"

DEFAULT_CONFIG = {
    "z_score_in": 2.0,
    "cap_gamba": 10000,
    "tp_z": 0.5,
    "cap_totale": 100000,
    "sl_z": 3.0,
    "sl_euro": -500
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)

if 'config' not in st.session_state:
    st.session_state.config = load_config()

CONFIG = st.session_state.config

# ============================================================
# DATABASE
# ============================================================
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS portafoglio (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker_long TEXT, prezzo_long REAL,
        ticker_short TEXT, prezzo_short REAL,
        z_in REAL, z_attuale REAL,
        stato TEXT DEFAULT 'APERTO',
        data_in TEXT, qty_long INTEGER, qty_short INTEGER)""")
    c.execute("""CREATE TABLE IF NOT EXISTS storico_chiusi (
        id TEXT PRIMARY KEY, pair TEXT,
        data_in TEXT, data_out TEXT, giorni INTEGER,
        z_in REAL, z_out REAL,
        qty_long INTEGER, p_in_long REAL, p_out_long REAL,
        qty_short INTEGER, p_in_short REAL, p_out_short REAL,
        pnl_lordo REAL, commissioni REAL, pnl_netto REAL, equity_cum REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS sp500_db (
        ticker TEXT PRIMARY KEY, market_cap REAL, perf_90g REAL,
        volume_avg REAL, beta REAL, settore TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist_intersettoriale (
        ticker1 TEXT, settore1 TEXT, ticker2 TEXT, settore2 TEXT,
        corr REAL, adf REAL, hedge_ratio REAL, half_life REAL, hurst REAL)""")
    conn.commit()
    return conn

# ============================================================
# FUNZIONI MATEMATICHE
# ============================================================
def calcola_pearson(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = min(len(x), len(y))
    if n < 2: return 0.0
    x, y = x[:n], y[:n]
    if np.std(x) == 0 or np.std(y) == 0: return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def calcola_pvalue(ratios):
    ratios = np.array(ratios, dtype=float)
    if len(ratios) < 3: return 1.0
    diffs = np.diff(ratios)
    lags = ratios[:-1]
    slope, _, _, _, _ = stats.linregress(lags, diffs)
    return min(float(np.exp(slope * 5)), 1.0)

def calcola_hurst(serie):
    try:
        serie = np.array(serie, dtype=float)
        if len(serie) < 10: return 0.5
        log_ret = np.log(serie[1:] / serie[:-1])
        log_ret = log_ret[np.isfinite(log_ret)]
        if len(log_ret) < 5: return 0.5
        n = len(log_ret)
        stdev = np.std(log_ret)
        if stdev == 0: return 0.5
        max_scroll = np.max(serie) - np.min(serie)
        if max_scroll <= 0: return 0.5
        return float(np.log(max_scroll / stdev) / np.log(n))
    except:
        return 0.5

def calcola_half_life(spread):
    try:
        spread = np.array(spread, dtype=float)
        if len(spread) < 5: return 99.0
        x, y = spread[:-1], spread[1:] - spread[:-1]
        slope, _, _, _, _ = stats.linregress(x, y)
        if slope >= 0: return 99.0
        return float(np.clip(-np.log(2) / slope, 1, 99))
    except:
        return 99.0

def calcola_adf(spread):
    try:
        spread = np.array(spread, dtype=float)
        if len(spread) < 10: return 0.0
        diffs, lags = np.diff(spread), spread[:-1]
        slope, _, _, _, se = stats.linregress(lags, diffs)
        if se == 0: return 0.0
        return float(slope / se)
    except:
        return 0.0

def calcola_hedge_ratio(y, x):
    y, x = np.array(y, dtype=float), np.array(x, dtype=float)
    n = min(len(y), len(x))
    if n < 5: return 1.0
    y, x = y[:n], x[:n]
    slope, _, _, _, _ = stats.linregress(x, y)
    return float(slope) if np.isfinite(slope) else 1.0

# ============================================================
# FUNZIONI DATI
# ============================================================
@st.cache_data(ttl=3600)
def scarica_sp500():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'}
        resp = requests.get(url, headers=headers, timeout=30)
        tables = pd.read_html(resp.text)
        df = tables[0]
        tickers = [t.replace('.', '-') for t in df['Symbol'].tolist()]
        settori = df['GICS Sector'].tolist()
        return tickers, settori
    except:
        return [
            "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","BRK-B","UNH","JNJ",
            "XOM","JPM","V","PG","MA","HD","CVX","MRK","ABBV","LLY",
            "PEP","KO","COST","AVGO","TMO","MCD","WMT","CSCO","ACN","ABT",
            "DHR","LIN","NEE","TXN","PM","UNP","RTX","LOW","HON","AMGN",
            "IBM","QCOM","INTC","CAT","GS","BA","MMM","DIS","NKE","VZ"
        ], ["TECH"] * 50

@st.cache_data(ttl=1800)
def scarica_storico(tickers, giorni):
    try:
        data = yf.download(tickers, period=f"{giorni}d", progress=False)
        if data.empty:
            return {}
        result = {}
        for t in tickers:
            try:
                if len(tickers) == 1:
                    result[t] = data['Close'].dropna().tolist()
                else:
                    result[t] = data['Close'][t].dropna().tolist()
            except:
                result[t] = []
        return result
    except:
        return {}

@st.cache_data(ttl=300)
def scarica_prezzi_live(tickers):
    try:
        if len(tickers) == 1:
            data = yf.download(tickers, period="2d", progress=False)
            if data.empty: return {}
            return {tickers[0]: float(data['Close'].iloc[-1])}
        data = yf.download(tickers, period="2d", progress=False)
        if data.empty: return {}
        result = {}
        for t in tickers:
            try:
                result[t] = float(data['Close'][t].iloc[-1])
            except:
                result[t] = None
        return result
    except:
        return {}

# ============================================================
# SIDEBAR - NAVIGAZIONE
# ============================================================
st.sidebar.markdown("""
<div style='text-align:center; padding:10px;'>
    <h2>🚀 OMNI-PAIR</h2>
    <p style='color:gray;'>Pair Trading System v2026.1</p>
</div>
""", unsafe_allow_html=True)

pagina = st.sidebar.radio(
    "📌 Navigazione",
    [
        "🏠 Dashboard",
        "🔍 Scanner Settoriale",
        "⏳ Backtest Profondo",
        "➕ Apri Trade",
        "🏁 Chiudi Trade",
        "📊 Portafoglio",
        "💎 Storico Trade",
        "📈 OMNI MATRIX",
        "⚙️ Impostazioni"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption(f"💰 Cap/Gamba: ${CONFIG['cap_gamba']:,}")
st.sidebar.caption(f"📊 Z-Score In: {CONFIG['z_score_in']}")
st.sidebar.caption(f"🛑 SL Z: {CONFIG['sl_z']}")

# ============================================================
# PAGINA: DASHBOARD
# ============================================================
if pagina == "🏠 Dashboard":
    st.title("🏠 Dashboard OMNI-PAIR")
    
    conn = get_db()
    c = conn.cursor()
    
    # Statistiche
    c.execute("SELECT COUNT(*) FROM portafoglio WHERE stato='APERTO'")
    trade_aperti = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM storico_chiusi")
    trade_totali = c.fetchone()[0]
    
    c.execute("SELECT COALESCE(SUM(pnl_netto), 0) FROM storico_chiusi")
    pnl_totale = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM storico_chiusi WHERE pnl_netto > 0")
    trade_profit = c.fetchone()[0]
    
    win_rate = (trade_profit / trade_totali * 100) if trade_totali > 0 else 0
    
    c.execute("SELECT COALESCE(MAX(equity_cum), 0) FROM storico_chiusi")
    equity = c.fetchone()[0]
    
    conn.close()
    
    # Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📂 Trade Aperti", trade_aperti)
    col2.metric("📊 Trade Totali", trade_totali)
    col3.metric("🎯 Win Rate", f"{win_rate:.1f}%")
    col4.metric("💰 P/L Totale", f"${pnl_totale:,.2f}",
                delta=f"${pnl_totale:,.2f}" if pnl_totale != 0 else None)
    
    st.markdown("---")
    
    # Equity Curve
    if trade_totali > 0:
        conn = get_db()
        df_storico = pd.read_sql_query(
            "SELECT data_out, pnl_netto, equity_cum FROM storico_chiusi ORDER BY data_out",
            conn
        )
        conn.close()
        
        if not df_storico.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_storico['data_out'],
                y=df_storico['equity_cum'],
                mode='lines+markers',
                name='Equity',
                line=dict(color='#34a853', width=3),
                marker=dict(size=8)
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
            fig.update_layout(
                title="📈 Equity Curve",
                xaxis_title="Data",
                yaxis_title="Equity ($)",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👋 Benvenuto! Inizia inizializzando il database S&P500 dalla pagina **OMNI MATRIX**.")

# ============================================================
# PAGINA: SCANNER SETTORIALE
# ============================================================
elif pagina == "🔍 Scanner Settoriale":
    st.title("🔍 Scanner Settoriale")
    
    col1, col2 = st.columns(2)
    with col1:
        settore = st.selectbox(
            "Settore",
            ["TUTTI", "ENERGY", "MATERIALS", "INDUSTRIALS",
             "CONSUMER DISCRETIONARY", "CONSUMER STAPLES",
             "HEALTH CARE", "FINANCIALS", "INFORMATION TECHNOLOGY",
             "COMMUNICATION SERVICES", "UTILITIES", "REAL ESTATE"]
        )
    with col2:
        soglia_z = st.number_input("Soglia Z-Score", 1.0, 3.5, CONFIG['z_score_in'], 0.1)
    
    if st.button("🚀 Avvia Scanner", type="primary", use_container_width=True):
        conn = get_db()
        sp500 = pd.read_sql_query("SELECT * FROM sp500_db", conn)
        conn.close()
        
        if sp500.empty:
            st.error("⚠️ Database S&P500 vuoto! Vai su **OMNI MATRIX** per inizializzarlo.")
        else:
            if settore == "TUTTI":
                tickers = sp500['ticker'].tolist()[:60]
            else:
                tickers = sp500[sp500['settore'] == settore]['ticker'].tolist()[:40]
            
            if len(tickers) < 2:
                st.warning("⚠️ Pochi titoli nel settore selezionato")
            else:
                st.info(f"📈 Analisi su {len(tickers)} titoli...")
                progress = st.progress(0)
                
                storici = scarica_storico(tickers, 60)
                db_prezzi = {t: s for t, s in storici.items() if len(s) >= 25}
                validi = list(db_prezzi.keys())
                
                st.write(f"✅ {len(validi)} titoli con dati validi")
                
                segnali = []
                totale_coppie = len(validi) * (len(validi) - 1) // 2
                count = 0
                
                for i in range(len(validi)):
                    for j in range(i+1, len(validi)):
                        count += 1
                        if count % 100 == 0:
                            progress.progress(min(count / totale_coppie, 1.0))
                        
                        t_a, t_b = validi[i], validi[j]
                        s_a = np.array(db_prezzi[t_a][-60:])
                        s_b = np.array(db_prezzi[t_b][-60:])
                        min_len = min(len(s_a), len(s_b))
                        s_a, s_b = s_a[-min_len:], s_b[-min_len:]
                        
                        corr = calcola_pearson(s_a, s_b)
                        if corr < 0.70: continue
                        
                        ratios = s_a / s_b
                        p_val = calcola_pvalue(ratios)
                        if p_val > 0.10: continue
                        
                        slice30 = ratios[-31:-1]
                        media, dev = np.mean(slice30), np.std(slice30, ddof=1)
                        if dev == 0: continue
                        
                        z = (ratios[-1] - media) / dev
                        abs_z = abs(z)
                        
                        if abs_z > 1.5:
                            stelle = "⭐⭐⭐" if abs_z > 2.5 else ("⭐⭐" if abs_z > 2 else "⭐")
                            azione = f"BUY {t_a} / SELL {t_b}" if z < 0 else f"SELL {t_a} / BUY {t_b}"
                            segnali.append({
                                'Pair': f"{t_a}/{t_b}",
                                'Z-Score': round(z, 2),
                                'Corr': round(corr, 2),
                                'P-Val': round(p_val, 3),
                                'Conv': stelle,
                                'Azione': azione,
                                'Prezzo A': round(s_a[-1], 2),
                                'Prezzo B': round(s_b[-1], 2),
                                'abs_z': abs_z
                            })
                
                progress.progress(1.0)
                segnali.sort(key=lambda x: x['abs_z'], reverse=True)
                
                if segnali:
                    st.success(f"🎯 Trovati {len(segnali)} segnali!")
                    df_segnali = pd.DataFrame(segnali[:30])
                    df_segnali = df_segnali.drop(columns=['abs_z'])
                    st.dataframe(df_segnali, use_container_width=True, hide_index=True)
                else:
                    st.warning("⚠️ Nessun segnale trovato con i parametri attuali")

# ============================================================
# PAGINA: BACKTEST PROFONDO
# ============================================================
elif pagina == "⏳ Backtest Profondo":
    st.title("⏳ Backtest Profondo")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        t_a = st.text_input("Ticker A (LONG)", "AAPL").upper()
    with col2:
        t_b = st.text_input("Ticker B (SHORT)", "MSFT").upper()
    with col3:
        giorni = st.number_input("Giorni", 250, 2000, 1000, 50)
    with col4:
        soglia_z = st.number_input("Z-Score", 1.0, 3.5, CONFIG['z_score_in'], 0.1)
    
    if st.button("🚀 Avvia Backtest", type="primary", use_container_width=True):
        if not t_a or not t_b:
            st.error("⚠️ Inserisci entrambi i ticker")
        else:
            with st.spinner("⏳ Download dati e calcolo..."):
                storici = scarica_storico([t_a, t_b], giorni)
                serie_a = np.array(storici.get(t_a, []))
                serie_b = np.array(storici.get(t_b, []))
                
                if len(serie_a) < 50 or len(serie_b) < 50:
                    st.error("⚠️ Dati insufficienti")
                else:
                    min_len = min(len(serie_a), len(serie_b))
                    serie_a, serie_b = serie_a[-min_len:], serie_b[-min_len:]
                    ratios = serie_a / serie_b
                    
                    hurst = calcola_hurst(ratios)
                    hl = calcola_half_life(ratios)
                    adf = calcola_adf(ratios)
                    
                    budget = CONFIG['cap_gamba']
                    profitto, trade_tot, trade_win = 0.0, 0, 0
                    gain, loss, giorni_tot = 0.0, 0.0, 0
                    max_eq, max_dd, in_pos = 0.0, 0.0, False
                    equity_curve = []
                    
                    for i in range(30, len(ratios)):
                        finestra = ratios[i-30:i]
                        media, dev = np.mean(finestra), np.std(finestra, ddof=1)
                        if dev == 0:
                            equity_curve.append(profitto)
                            continue
                        z = (ratios[i] - media) / dev
                        if not in_pos:
                            if abs(z) > soglia_z:
                                in_pos = True
                                tipo = "LONG_A" if z < -soglia_z else "LONG_B"
                                z_ing, idx_ing = z, i
                                p_a_ing, p_b_ing = serie_a[i], serie_b[i]
                                qty_a = int(budget / p_a_ing) if p_a_ing > 0 else 0
                                qty_b = int(budget / p_b_ing) if p_b_ing > 0 else 0
                        else:
                            chiuso = False
                            if tipo == "LONG_A":
                                if z >= z_ing * 0.2 or z <= z_ing * 1.5:
                                    chiuso = True
                                    pnl = (serie_a[i] - p_a_ing) * qty_a + (p_b_ing - serie_b[i]) * qty_b
                            else:
                                if z <= z_ing * 0.2 or z >= z_ing * 1.5:
                                    chiuso = True
                                    pnl = (serie_b[i] - p_b_ing) * qty_b + (p_a_ing - serie_a[i]) * qty_a
                            if chiuso:
                                trade_tot += 1
                                profitto += pnl
                                giorni_tot += (i - idx_ing)
                                if pnl > 0:
                                    trade_win += 1
                                    gain += pnl
                                else:
                                    loss += abs(pnl)
                                if profitto > max_eq:
                                    max_eq = profitto
                                dd = max_eq - profitto
                                if dd > max_dd:
                                    max_dd = dd
                                in_pos = False
                        equity_curve.append(profitto)
                    
                    win_rate = trade_win / trade_tot if trade_tot > 0 else 0
                    
                    # Risultati
                    st.markdown("---")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("💰 Profitto", f"${profitto:,.2f}")
                    c2.metric("📊 Trade", f"{trade_tot} ({trade_win}W)")
                    c3.metric("🎯 Win Rate", f"{win_rate*100:.1f}%")
                    c4.metric("📉 Max DD", f"${max_dd:,.2f}")
                    
                    c5, c6, c7 = st.columns(3)
                    c5.metric("Hurst", f"{hurst:.3f}",
                              delta="✅" if hurst < 0.55 else "⚠️",
                              delta_color="normal")
                    c6.metric("Half-Life", f"{hl:.0f}gg",
                              delta="✅" if hl < 50 else "⚠️",
                              delta_color="normal")
                    c7.metric("ADF Stat", f"{adf:.3f}",
                              delta="✅" if adf < -2.5 else "⚠️",
                              delta_color="normal")
                    
                    # Equity Curve
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=equity_curve, mode='lines',
                        name='Equity',
                        line=dict(color='#34a853', width=2),
                        fill='tozeroy', fillcolor='rgba(52,168,83,0.1)'
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
                    fig.update_layout(
                        title=f"📈 Equity Curve - {t_a}/{t_b}",
                        xaxis_title="Giorni", yaxis_title="P/L ($)",
                        template="plotly_white", height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Ratio Chart
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        y=ratios, mode='lines',
                        name='Ratio', line=dict(color='#1a73e8', width=1)
                    ))
                    media_ratio = np.mean(ratios)
                    fig2.add_hline(y=media_ratio, line_dash="solid", line_color="green")
                    fig2.add_hline(y=media_ratio + soglia_z * np.std(ratios), line_dash="dash", line_color="red")
                    fig2.add_hline(y=media_ratio - soglia_z * np.std(ratios), line_dash="dash", line_color="red")
                    fig2.update_layout(
                        title=f"📊 Price Ratio - {t_a}/{t_b}",
                        template="plotly_white", height=300
                    )
                    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# PAGINA: APRI TRADE
# ============================================================
elif pagina == "➕ Apri Trade":
    st.title("➕ Apri Nuovo Trade")
    
    pair_input = st.text_input("Inserisci coppia (es. KO/PEP)", "KO/PEP").upper()
    
    if "/" in pair_input:
        t1, t2 = [x.strip() for x in pair_input.split("/")]
        
        with st.spinner("⏳ Download prezzi live..."):
            prezzi = scarica_prezzi_live([t1, t2])
            p1 = prezzi.get(t1)
            p2 = prezzi.get(t2)
        
        if p1 and p2:
            st.success(f"💰 {t1}: ${p1:.2f} | {t2}: ${p2:.2f}")
            
            direzione = st.radio("Direzione", [
                f"BUY {t1} / SELL {t2}",
                f"SELL {t1} / BUY {t2}"
            ])
            
            if direzione.startswith("BUY"):
                t_long, t_short = t1, t2
                p_long, p_short = p1, p2
            else:
                t_long, t_short = t2, t1
                p_long, p_short = p2, p1
            
            col1, col2 = st.columns(2)
            with col1:
                p_long_in = st.number_input(f"Prezzo LONG {t_long}", 0.01, 99999.0, float(p_long), 0.01)
            with col2:
                p_short_in = st.number_input(f"Prezzo SHORT {t_short}", 0.01, 99999.0, float(p_short), 0.01)
            
            budget = CONFIG['cap_gamba']
            qty_l = int(budget / p_long_in)
            qty_s = int(budget / p_short_in)
            
            st.info(f"💵 Budget: ${budget} | Qty LONG: {qty_l} | Qty SHORT: {qty_s}")
            
            z_in = st.number_input("Z-Score ingresso", -5.0, 5.0, CONFIG['z_score_in'], 0.01)
            
            if st.button("✅ Conferma Apertura", type="primary", use_container_width=True):
                conn = get_db()
                c = conn.cursor()
                c.execute("""INSERT INTO portafoglio 
                    (ticker_long, prezzo_long, ticker_short, prezzo_short, z_in, data_in, qty_long, qty_short)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (t_long, p_long_in, t_short, p_short_in, z_in,
                     datetime.now().isoformat(), qty_l, qty_s))
                conn.commit()
                trade_id = c.lastrowid
                conn.close()
                st.success(f"✅ Trade #{trade_id} aperto! LONG {t_long}@{p_long_in} / SHORT {t_short}@{p_short_in}")
                st.balloons()
        else:
            st.error("⚠️ Impossibile ottenere i prezzi. Verifica i ticker.")
    else:
        st.warning("⚠️ Formato non valido. Usa T1/T2")

# ============================================================
# PAGINA: CHIUDI TRADE
# ============================================================
elif pagina == "🏁 Chiudi Trade":
    st.title("🏁 Chiudi Trade")
    
    conn = get_db()
    trade_aperti = pd.read_sql_query(
        "SELECT * FROM portafoglio WHERE stato='APERTO'", conn
    )
    conn.close()
    
    if trade_aperti.empty:
        st.info("⚠️ Nessun trade aperto")
    else:
        st.dataframe(trade_aperti, use_container_width=True, hide_index=True)
        
        trade_id = st.number_input("ID Trade da chiudere", 1, 9999, 1)
        
        if st.button("🏁 Chiudi Trade", type="primary", use_container_width=True):
            conn = get_db()
            c = conn.cursor()
            c.execute("SELECT * FROM portafoglio WHERE id=?", (trade_id,))
            t = c.fetchone()
            
            if not t:
                st.error(f"⚠️ Trade #{trade_id} non trovato")
            else:
                prezzi = scarica_prezzi_live([t['ticker_long'], t['ticker_short']])
                p_l = prezzi.get(t['ticker_long'], t['prezzo_long'])
                p_s = prezzi.get(t['ticker_short'], t['prezzo_short'])
                
                pnl_lordo = ((p_l - t['prezzo_long']) * t['qty_long'] +
                             (t['prezzo_short'] - p_s) * t['qty_short'])
                
                st.info(f"📊 P/L Lordo: ${pnl_lordo:,.2f}")
                
                data_in = datetime.fromisoformat(t['data_in'])
                giorni = max(1, (datetime.now() - data_in).days)
                pair = f"{t['ticker_long']}/{t['ticker_short']}"
                
                c.execute("SELECT COALESCE(MAX(equity_cum), 0) FROM storico_chiusi")
                eq_prec = c.fetchone()[0]
                equity = eq_prec + pnl_lordo
                
                c.execute("""INSERT INTO storico_chiusi 
                    (id, pair, data_in, data_out, giorni, z_in, z_out, qty_long, p_in_long, p_out_long,
                     qty_short, p_in_short, p_out_short, pnl_lordo, commissioni, pnl_netto, equity_cum)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (pair, pair, t['data_in'], datetime.now().isoformat(), giorni, t['z_in'], 0.0,
                     t['qty_long'], t['prezzo_long'], p_l, t['qty_short'], t['prezzo_short'],
                     p_s, pnl_lordo, 0.0, pnl_lordo, equity))
                c.execute("DELETE FROM portafoglio WHERE id=?", (trade_id,))
                conn.commit()
                conn.close()
                
                st.success(f"✅ Trade chiuso! P/L: ${pnl_lordo:,.2f} | Equity: ${equity:,.2f}")
                st.balloons()

# ============================================================
# PAGINA: PORTAFOGLIO
# ============================================================
elif pagina == "📊 Portafoglio":
    st.title("📊 Portafoglio Aperto")
    
    conn = get_db()
    trade = pd.read_sql_query(
        "SELECT * FROM portafoglio WHERE stato='APERTO'", conn
    )
    conn.close()
    
    if trade.empty:
        st.info("⚠️ Nessun trade aperto")
    else:
        # Aggiorna prezzi live
        tickers = list(set(trade['ticker_long'].tolist() + trade['ticker_short'].tolist()))
        prezzi = scarica_prezzi_live(tickers)
        
        pnl_totale = 0
        for idx, row in trade.iterrows():
            p_l = prezzi.get(row['ticker_long'], row['prezzo_long'])
            p_s = prezzi.get(row['ticker_short'], row['prezzo_short'])
            if p_l and p_s:
                pnl = ((p_l - row['prezzo_long']) * row['qty_long'] +
                       (row['prezzo_short'] - p_s) * row['qty_short'])
                pnl_totale += pnl
        
        st.metric("💰 P/L Totale Portafoglio", f"${pnl_totale:,.2f}")
        st.dataframe(trade, use_container_width=True, hide_index=True)

# ============================================================
# PAGINA: STORICO
# ============================================================
elif pagina == "💎 Storico Trade":
    st.title("💎 Storico Trade Chiusi")
    
    conn = get_db()
    storico = pd.read_sql_query(
        "SELECT * FROM storico_chiusi ORDER BY data_in DESC", conn
    )
    conn.close()
    
    if storico.empty:
        st.info("⚠️ Nessun trade chiuso")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("📊 Totale Trade", len(storico))
        c2.metric("🎯 Win Rate",
                  f"{len(storico[storico['pnl_netto']>0])/len(storico)*100:.1f}%")
        c3.metric("💰 P/L Totale", f"${storico['pnl_netto'].sum():,.2f}")
        
        st.dataframe(storico, use_container_width=True, hide_index=True)
        
        # Grafico P/L per trade
        fig = go.Figure()
        colors = ['#34a853' if x > 0 else '#ea4335' for x in storico['pnl_netto']]
        fig.add_trace(go.Bar(
            x=storico['pair'],
            y=storico['pnl_netto'],
            marker_color=colors,
            name='P/L Netto'
        ))
        fig.update_layout(
            title="📊 P/L per Trade",
            xaxis_title="Pair", yaxis_title="P/L ($)",
            template="plotly_white", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGINA: OMNI MATRIX
# ============================================================
elif pagina == "📈 OMNI MATRIX":
    st.title("📈 OMNI MATRIX - Ricerca & Analisi")
    
    st.markdown("""
    L'OMNI MATRIX analizza il mercato in **3 fasi**:
    1. **Shortlist** - Seleziona i migliori titoli S&P500
    2. **Correlazioni** - Trova coppie intersettoriali correlate
    3. **Cointegrazione** - Test ADF, Hurst, Half-Life
    """)
    
    if st.button("🚀 Avvia OMNI MATRIX (Fase 1: Shortlist)", type="primary", use_container_width=True):
        with st.spinner("⏳ Scarico lista S&P500 da Wikipedia..."):
            tickers, settori = scarica_sp500()
            st.write(f"✅ Trovati {len(tickers)} ticker")
        
        with st.spinner("⏳ Analisi fondamentali (5-10 min)..."):
            progress = st.progress(0)
            risultati = []
            for i, t in enumerate(tickers[:100]):
                if i % 10 == 0:
                    progress.progress(i / 100)
                try:
                    tk = yf.Ticker(t)
                    info = tk.info
                    risultati.append({
                        'ticker': t,
                        'market_cap': (info.get('marketCap') or 0) / 1e9,
                        'beta': info.get('beta', 1.0),
                        'volume_avg': info.get('averageVolume', 0),
                        'settore': settori[i] if i < len(settori) else 'N/A',
                        'perf_90g': 0
                    })
                except:
                    pass
            
            progress.progress(1.0)
            
            conn = get_db()
            c = conn.cursor()
            c.execute("DELETE FROM sp500_db")
            for item in risultati:
                c.execute("INSERT INTO sp500_db VALUES (?,?,?,?,?,?)",
                    (item['ticker'], item['market_cap'], item['perf_90g'],
                     item['volume_avg'], item['beta'], item['settore']))
            conn.commit()
            conn.close()
            
            st.success(f"✅ Database inizializzato con {len(risultati)} titoli!")
            st.dataframe(pd.DataFrame(risultati).head(20), use_container_width=True, hide_index=True)

# ============================================================
# PAGINA: IMPOSTAZIONI
# ============================================================
elif pagina == "⚙️ Impostazioni":
    st.title("⚙️ Impostazioni")
    
    st.markdown("### Parametri di Trading")
    
    col1, col2 = st.columns(2)
    with col1:
        new_z = st.number_input("Z-Score Ingresso", 1.0, 3.5, CONFIG['z_score_in'], 0.1)
        new_cap = st.number_input("Capitale per Gamba ($)", 1000, 1000000, CONFIG['cap_gamba'], 1000)
        new_tp = st.number_input("TP Z-Score", 0.0, 2.0, CONFIG['tp_z'], 0.1)
    with col2:
        new_sl_z = st.number_input("SL Z-Score", 1.0, 5.0, CONFIG['sl_z'], 0.1)
        new_sl_euro = st.number_input("SL Monetario ($)", -10000, 0, CONFIG['sl_euro'], 100)
        new_cap_tot = st.number_input("Capitale Totale ($)", 10000, 10000000, CONFIG['cap_totale'], 10000)
    
    if st.button("💾 Salva Impostazioni", type="primary", use_container_width=True):
        CONFIG['z_score_in'] = new_z
        CONFIG['cap_gamba'] = new_cap
        CONFIG['tp_z'] = new_tp
        CONFIG['sl_z'] = new_sl_z
        CONFIG['sl_euro'] = new_sl_euro
        CONFIG['cap_totale'] = new_cap_tot
        save_config(CONFIG)
        st.session_state.config = CONFIG
        st.success("✅ Impostazioni salvate!")
        st.rerun()
