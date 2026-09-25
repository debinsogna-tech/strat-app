"""
stratified_wealth_protection_app.py
Streamlit web interface for the stratified wealth-protection framework.
"""

import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

# --- Core Classes & Logic (Same as your script) ---
@dataclass
class Stratum:
    name: str
    weight: float
    n_units: int
    std: float
    cost: float = 1.0
    loss_paths: np.ndarray = None
    description: str = ""

@dataclass
class AllocationResult:
    stratum_name: str
    n_sample: int
    weight: float
    std: float
    cost: float

@dataclass
class RiskEstimate:
    mean_loss: float
    cvar: float
    var: float
    std_error: float
    stratum_contributions: Dict[str, float]

def neyman_allocation(strata: List[Stratum], total_n: int, use_costs: bool = True) -> List[AllocationResult]:
    if total_n <= 0:
        raise ValueError("total_n must be positive")
    numerators = []
    for s in strata:
        scale = s.n_units * s.std
        if use_costs:
            scale /= np.sqrt(max(s.cost, 1e-12))
        numerators.append(scale)
    total = sum(numerators)
    if total <= 0:
        numerators = [s.n_units for s in strata]
        total = sum(numerators)
    raw = [total_n * (num / total) for num in numerators]
    floors = [int(np.floor(r)) for r in raw]
    remainders = [r - f for r, f in zip(raw, floors)]
    leftover = total_n - sum(floors)
    order = np.argsort(remainders)[::-1]
    for i in range(leftover):
        floors[order[i]] += 1
    return [AllocationResult(s.name, max(1, n), s.weight, s.std, s.cost) for s, n in zip(strata, floors)]

def stratified_sample(strata: List[Stratum], allocation: List[AllocationResult], rng: np.random.Generator) -> Dict[str, np.ndarray]:
    samples = {}
    alloc_map = {a.stratum_name: a.n_sample for a in allocation}
    for s in strata:
        n = alloc_map[s.name]
        paths = np.asarray(s.loss_paths)
        idx = rng.choice(paths.shape[0] if paths.ndim > 1 else len(paths), size=min(n, paths.shape[0] if paths.ndim > 1 else len(paths)), replace=False)
        samples[s.name] = paths[idx]
    return samples

def estimate_cvar(losses: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
    losses = np.asarray(losses).ravel()
    if len(losses) == 0:
        return np.nan, np.nan
    var = np.quantile(losses, alpha)
    cvar = losses[losses >= var].mean() if np.any(losses >= var) else var
    return float(var), float(cvar)

def stratified_risk_estimate(strata: List[Stratum], samples: Dict[str, np.ndarray], alpha: float = 0.95) -> RiskEstimate:
    mean_loss = 0.0
    var_sum = 0.0
    contributions = {}
    all_weighted_losses = []
    for s in strata:
        w = s.weight
        samp = samples[s.name].ravel()
        m_h = samp.mean()
        s2_h = samp.var(ddof=1) if len(samp) > 1 else 0.0
        n_h = len(samp)
        mean_loss += w * m_h
        var_sum += (w ** 2) * (s2_h / n_h)
        contributions[s.name] = w * m_h
        all_weighted_losses.append(np.full_like(samp, w) * samp)
    pooled = np.concatenate(all_weighted_losses) if all_weighted_losses else np.array([])
    var, cvar = estimate_cvar(pooled, alpha)
    return RiskEstimate(mean_loss, cvar, var, np.sqrt(var_sum), contributions)

def suggest_protections(risk: RiskEstimate, strata: List[Stratum], cvar_budget: float) -> Dict[str, str]:
    suggestions = {}
    total_cvar = risk.cvar
    for s in strata:
        contrib = risk.stratum_contributions.get(s.name, 0.0)
        share = contrib / total_cvar if total_cvar != 0 else 0.0
        if "equity" in s.name.lower():
            suggestions[s.name] = f"High share ({share:.1%}). Consider put spreads / collars."
        elif "fixed" in s.name.lower():
            suggestions[s.name] = "Monitor duration & credit. Add cash buffer if needed."
        elif "real" in s.name.lower():
            suggestions[s.name] = "Inflation-sensitive. Evaluate TIPS or real assets."
        elif "private" in s.name.lower():
            suggestions[s.name] = "Illiquidity risk. Size secondary-sale reserve."
        else:
            suggestions[s.name] = "Longevity/sequence risk. Consider deferred annuity."
    
    if total_cvar > cvar_budget:
        suggestions["AGGREGATE"] = f"⚠️ CVaR ({total_cvar:.2%}) exceeds budget ({cvar_budget:.2%})."
    else:
        suggestions["AGGREGATE"] = "✅ Aggregate CVaR within budget."
    return suggestions

def make_synthetic_strata(n_scenarios: int = 5000, seed: int = 123) -> List[Stratum]:
    rng = np.random.default_rng(seed)
    equity_losses = - (rng.standard_t(df=5, size=(800, n_scenarios)) * 0.18 / np.sqrt(252)).mean(axis=1)
    bond_losses = - rng.normal(0.03, 0.06, size=(400, n_scenarios)).mean(axis=1)
    real_losses = - rng.normal(0.04, 0.12, size=(250, n_scenarios)).mean(axis=1)
    priv_losses = - (rng.standard_t(df=3, size=(150, n_scenarios)) * 0.22).mean(axis=1)
    long_losses = rng.lognormal(mean=0.02, sigma=0.25, size=100)
    
    return [
        Stratum("Liquid Risk Assets (Equities)", 0.45, 800, float(equity_losses.std()), 1.0, equity_losses),
        Stratum("Defensive Fixed Income", 0.25, 400, float(bond_losses.std()), 0.8, bond_losses),
        Stratum("Real Assets & Inflation Hedges", 0.15, 250, float(real_losses.std()), 1.5, real_losses),
        Stratum("Private & Alternatives", 0.10, 150, float(priv_losses.std()), 3.0, priv_losses),
        Stratum("Longevity / Human Capital", 0.05, 100, float(long_losses.std()), 2.0, long_losses),
    ]

# --- Streamlit UI Layout ---
st.title("🛡️ Stratified Wealth-Protection Framework")
st.markdown("Run modular stratified-random-sampling and risk analysis directly from your browser.")

# Sidebar controls for iPad touch input
st.sidebar.header("Simulation Parameters")
total_sample_size = st.sidebar.slider("Total Sample Size", 500, 5000, 1500, 100)
alpha = st.sidebar.selectbox("Confidence Level (Alpha)", [0.90, 0.95, 0.99], index=1)
cvar_budget = st.sidebar.slider("CVaR Budget", 0.01, 0.15, 0.06, 0.01)
seed = st.sidebar.number_input("Random Seed", value=42)

# Run pipeline
strata = make_synthetic_strata(seed=int(seed))
rng = np.random.default_rng(int(seed))
allocation = neyman_allocation(strata, total_sample_size)
samples = stratified_sample(strata, allocation, rng)
risk = stratified_risk_estimate(strata, samples, alpha=alpha)
protections = suggest_protections(risk, strata, cvar_budget=cvar_budget)

# Display Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean Loss", f"{risk.mean_loss:.4f}")
col2.metric("VaR", f"{risk.var:.4f}")
col3.metric("CVaR", f"{risk.cvar:.4f}")
col4.metric("Std Error", f"{risk.std_error:.4f}")

st.divider()

# Display Allocation Table
st.subheader("📊 Neyman Allocation")
alloc_data = [{"Stratum": a.stratum_name, "Samples": a.n_sample, "Weight": a.weight, "Std Dev": f"{a.std:.4f}", "Cost": a.cost} for a in allocation]
st.table(alloc_data)

# Display Protection Suggestions
st.subheader("💡 Protection Suggestions")
for k, v in protections.items():
    if k == "AGGREGATE":
        st.info(f"**{k}**: {v}")
    else:
        st.write(f"**{k}**: {v}")