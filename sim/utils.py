import numpy as np
import pandas as pd

def future_value_schedule(P0, contrib_m, rate_am, months, fee_am=0.0, shock_vol_am=None, seed=42):
    rng = np.random.default_rng(seed) if shock_vol_am is not None else None
    saldo = P0
    rows = []
    for m in range(1, months+1):
        r = rate_am
        if shock_vol_am is not None:
            r = (1 + r) * np.exp(rng.normal(loc=0.0, scale=shock_vol_am)) - 1
        r_eff = r - fee_am
        saldo = saldo * (1 + r_eff) + contrib_m
        rows.append((m, saldo))
    return pd.DataFrame(rows, columns=["Mês", "Saldo"])

def gbm_paths(mu_am, cov_am, weights, S0=1.0, months=120, n_paths=2000, seed=123):
    rng = np.random.default_rng(seed)
    mu_p = np.dot(weights, mu_am)
    var_p = weights @ cov_am @ weights
    sigma_p = np.sqrt(var_p)

    S = np.zeros((months+1, n_paths))
    S[0, :] = S0
    for t in range(1, months+1):
        z = rng.standard_normal(n_paths)
        growth = np.exp((mu_p - 0.5*sigma_p**2) + sigma_p * z)
        S[t, :] = S[t-1, :] * growth
    return S

def portfolio_stats(paths, risk_free=0.06):
    rets = paths[1:, :] / paths[:-1, :] - 1.0
    mean_m = rets.mean()
    std_m = rets.std()
    er_aa = (1 + mean_m) ** 12 - 1
    vol_aa = std_m * np.sqrt(12)
    sharpe = (er_aa - risk_free) / (vol_aa + 1e-9)
    return {
        "retorno_esperado_aa": float(er_aa),
        "volatilidade_aa": float(vol_aa),
        "sharpe_aproximado": float(sharpe)
    }

def run_guided_vs_chaotic_scenarios(P0, aporte_m, anos, ret_aa, vol_aa, fee_aa, seed=777):
    months = int(anos * 12)
    rate_am = (1 + ret_aa) ** (1/12) - 1
    vol_am = vol_aa / np.sqrt(12)
    fee_am_caos = (1 + 0.02) ** (1/12) - 1
    fee_am_guiado = (1 + fee_aa) ** (1/12) - 1

    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, vol_am, size=months)
    market = np.exp((rate_am - 0.5*vol_am**2) + shocks) - 1

    saldo_c = P0
    rows_c = []
    for m in range(1, months+1):
        timing_bias = -0.5 * market[m-1]
        mult = np.clip(rng.uniform(0.0, 1.5) + timing_bias, 0.0, 1.5)
        aporte_var = aporte_m * mult
        if rng.random() < 0.08:
            saque = min(saldo_c * rng.uniform(0.01, 0.05), saldo_c)
        else:
            saque = 0.0
        r_eff = market[m-1] - fee_am_caos
        saldo_c = (saldo_c - saque) * (1 + r_eff) + aporte_var
        rows_c.append((m, saldo_c))
    df_caos = pd.DataFrame(rows_c, columns=["Mês", "Saldo"])

    saldo_g = P0
    rows_g = []
    for m in range(1, months+1):
        r_eff = market[m-1] - fee_am_guiado
        saldo_g = saldo_g * (1 + r_eff) + aporte_m
        rows_g.append((m, saldo_g))
    df_guiado = pd.DataFrame(rows_g, columns=["Mês", "Saldo"])

    return df_caos, df_guiado
