import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sim.utils import (
    future_value_schedule,
    gbm_paths,
    portfolio_stats,
    run_guided_vs_chaotic_scenarios
)

st.set_page_config(page_title="Simulador de Investimentos — Ciclo 4", layout="wide")

st.title("📈 Simulador de Investimentos — Educação Financeira (Ciclo 4)")
st.markdown("""
Este simulador foi pensado para o **Ciclo 4** como um produto educacional que conecta **lógica matemática** à **realidade financeira**.
Ele inclui:
- **Juros compostos** (valor do dinheiro no tempo)
- **Carteiras, risco e retorno** (volatilidade, CAPM básico)
- **Cenário guiado vs. caótico** (antes/depois de aprender os conceitos)

> **Objetivo:** promover educação financeira acessível e prática, com visualizações e interação em tempo real.
""")

st.sidebar.header("Configurações Gerais")
currency = st.sidebar.selectbox("Moeda para exibição", ["R$", "US$"], index=0)
risk_free = st.sidebar.number_input("Taxa livre de risco (a.a., %)", value=6.0, step=0.1)

tab1, tab2, tab3 = st.tabs(["🔢 Juros Compostos", "📊 Carteira & Risco", "🧭 Guiado vs. Caótico"])

with tab1:
    st.subheader("🔢 Simulador de Juros Compostos")
    col_a, col_b = st.columns(2)

    with col_a:
        P0 = st.number_input("Capital inicial", min_value=0.0, value=1000.0, step=100.0)
        contrib_m = st.number_input("Aporte mensal", min_value=0.0, value=200.0, step=50.0)
        years = st.number_input("Tempo (anos)", min_value=0, value=10, step=1)
    with col_b:
        rate_aa = st.number_input("Taxa esperada (a.a., %)", value=10.0, step=0.5)
        vol_aa = st.number_input("Volatilidade anual (opcional, %)", value=0.0, step=0.5)
        fee_aa = st.number_input("Taxa de administração (a.a., %, opcional)", value=0.0, step=0.1)

    months = years * 12
    rate_am = (1 + rate_aa/100.0) ** (1/12) - 1
    fee_am = (1 + fee_aa/100.0) ** (1/12) - 1
    vol_am = (vol_aa/100.0) / np.sqrt(12) if vol_aa > 0 else 0.0

    df = future_value_schedule(P0, contrib_m, rate_am, months, fee_am, vol_am if vol_aa>0 else None)
    st.dataframe(df.head(12))

    fig1 = plt.figure()
    plt.plot(df["Mês"], df["Saldo"])
    plt.xlabel("Mês")
    plt.ylabel(f"Saldo ({currency})")
    plt.title("Evolução do Capital (juros compostos)")
    st.pyplot(fig1)

    total_aportado = (contrib_m * months) + P0
    st.metric("Total aportado", f"{currency} {total_aportado:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    st.metric("Saldo final", f"{currency} {df['Saldo'].iloc[-1]:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

with tab2:
    st.subheader("📊 Carteira, Risco & CAPM (básico)")
    st.markdown("Monte até **3 ativos** com retorno e risco anuais. Ajuste os pesos e, opcionalmente, uma correlação média.")

    n_assets = st.number_input("Número de ativos", min_value=1, max_value=3, value=3, step=1)
    names, exp_ret_aa, vol_aa_list, weights = [], [], [], []

    for i in range(int(n_assets)):
        st.markdown(f"**Ativo {i+1}**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            names.append(st.text_input(f"Nome {i+1}", value=f"Ativo {i+1}"))
        with c2:
            exp_ret_aa.append(st.number_input(f"Retorno esperado (a.a., %) {i+1}", value=10.0, step=0.5, key=f"ret{i}"))
        with c3:
            vol_aa_list.append(st.number_input(f"Volatilidade (a.a., %) {i+1}", value=20.0, step=0.5, key=f"vol{i}"))
        with c4:
            weights.append(st.slider(f"Peso {i+1}", 0.0, 1.0, 1.0/float(n_assets), 0.01, key=f"w{i}"))

    w = np.array(weights, dtype=float)
    if w.sum() == 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()

    corr = st.slider("Correlação média entre os ativos", -0.9, 0.95, 0.2, 0.05)
    years_p = st.number_input("Horizonte (anos)", min_value=1, value=10, step=1)
    n_paths = st.number_input("Caminhos Monte Carlo", min_value=100, max_value=10000, value=2000, step=100)

    mu_aa = np.array(exp_ret_aa)/100.0
    vol_aa_arr = np.array(vol_aa_list)/100.0
    mu_am = (1 + mu_aa) ** (1/12) - 1
    vol_am_arr = vol_aa_arr / np.sqrt(12)

    if len(vol_am_arr) == 1:
        cov = np.array([[vol_am_arr[0]**2]])
    else:
        V = np.diag(vol_am_arr)
        R = np.ones((len(vol_am_arr), len(vol_am_arr))) * corr
        np.fill_diagonal(R, 1.0)
        cov = V @ R @ V

    horizon_m = int(years_p*12)
    S0 = 1.0
    paths = gbm_paths(mu_am, cov, w, S0=S0, months=horizon_m, n_paths=int(n_paths))

    stats = portfolio_stats(paths, risk_free=risk_free/100.0)
    st.write("**Estatísticas da carteira (simulada):**")
    st.json(stats)

    fig2 = plt.figure()
    sample = min(50, paths.shape[1])
    for j in range(sample):
        plt.plot(paths[:, j])
    plt.xlabel("Mês")
    plt.ylabel("Valor da carteira (inicial=1.0)")
    plt.title("Amostra de caminhos simulados (GBM)")
    st.pyplot(fig2)

    fig3 = plt.figure()
    plt.hist(paths[-1, :], bins=30)
    plt.xlabel("Valor final")
    plt.ylabel("Frequência")
    plt.title("Distribuição dos valores finais")
    st.pyplot(fig3)

with tab3:
    st.subheader("🧭 Cenário: Caótico (antes) vs. Guiado (depois)")
    st.markdown("""
    Compare decisões **sem orientação matemática** (caóticas) com um plano **disciplinado** (guiado).
    """)
    c1, c2 = st.columns(2)
    with c1:
        P0_c = st.number_input("Capital inicial (ambos)", min_value=0.0, value=2000.0, step=100.0)
        aporte_c = st.number_input("Aporte mensal (guiado)", min_value=0.0, value=300.0, step=50.0)
        anos_c = st.number_input("Tempo (anos, ambos)", min_value=1, value=8, step=1)
    with c2:
        ret_aa_c = st.number_input("Retorno base (a.a., %)", value=9.0, step=0.5)
        vol_aa_c = st.number_input("Volatilidade (a.a., %)", value=18.0, step=0.5)
        fee_aa_c = st.number_input("Taxa de administração (a.a., %, guiado)", value=0.3, step=0.1)

    df_caos, df_guiado = run_guided_vs_chaotic_scenarios(
        P0=P0_c,
        aporte_m=aporte_c,
        anos=anos_c,
        ret_aa=ret_aa_c/100.0,
        vol_aa=vol_aa_c/100.0,
        fee_aa=fee_aa_c/100.0
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cenário Caótico (antes de aprender):**")
        fig4 = plt.figure()
        plt.plot(df_caos["Mês"], df_caos["Saldo"])
        plt.xlabel("Mês")
        plt.ylabel(f"Saldo ({currency})")
        plt.title("Evolução do capital — Caótico")
        st.pyplot(fig4)
        st.dataframe(df_caos.tail(5))

    with col2:
        st.markdown("**Cenário Guiado (depois de aprender):**")
        fig5 = plt.figure()
        plt.plot(df_guiado["Mês"], df_guiado["Saldo"])
        plt.xlabel("Mês")
        plt.ylabel(f"Saldo ({currency})")
        plt.title("Evolução do capital — Guiado")
        st.pyplot(fig5)
        st.dataframe(df_guiado.tail(5))

    st.markdown("### Comparativo final")
    colf1, colf2, colf3 = st.columns(3)
    colf1.metric("Saldo final — Caótico", f"{currency} {df_caos['Saldo'].iloc[-1]:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    colf2.metric("Saldo final — Guiado", f"{currency} {df_guiado['Saldo'].iloc[-1]:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    delta = df_guiado['Saldo'].iloc[-1] - df_caos['Saldo'].iloc[-1]
    colf3.metric("Diferença (Guiado − Caótico)", f"{currency} {delta:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

st.markdown("---")
st.caption("Protótipo educacional — não é recomendação de investimento. Autores do projeto: Grupo Ciclo 4.")
