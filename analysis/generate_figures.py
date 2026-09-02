from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


try:
    trapezoid = np.trapezoid
except AttributeError:  # NumPy 1.x compatibility
    trapezoid = np.trapz


# Resolve paths relative to the repository so the script works on Linux,
# macOS, Windows, and in CI without editing machine-specific directories.
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Illustrative baseline. Most market and insurance parameters follow the
# numerical scale used by Li, Rong and Zhao (2015). Gamma and the investment
# caps are chosen so that both compact-control switches occur inside [0,T].
T = 10.0
r = 0.03
gamma = 0.35
mu1, mu2 = 0.09, 0.10
sigma1, sigma2 = 0.25, 0.30
lambda1, lambda2 = 1.20, 1.30
b1, b2 = 1.50, 1.80
pi1_bar, pi2_bar = 1.50, 1.90
a1, a2 = 0.50, 0.60

m1, m2 = mu1 - r, mu2 - r
A2 = m2 / (gamma * sigma2**2)
B2 = lambda2 / (gamma * b2**2)


def clipped_switch_time(candidate_at_T, bound):
    """Boundary-entry time, clipped to [0,T]."""
    raw = T - np.log(candidate_at_T / bound) / r
    return float(np.clip(raw, 0.0, T))


t_pi = clipped_switch_time(A2, pi2_bar)
t_p = clipped_switch_time(B2, 1.0)

t = np.linspace(0.0, T, 1001)
discount = np.exp(-r * (T - t))
pi2_hat = A2 * discount
p2_hat = B2 * discount
pi2 = np.minimum(pi2_bar, pi2_hat)
p2 = np.minimum(1.0, p2_hat)
pi1 = np.full_like(t, pi1_bar)
p1 = np.ones_like(t)

# Equilibrium value-function coefficient.  The baseline has D=0, but the
# expression is kept in its general form for reproducibility.
phi = np.exp(r * (T - t))
D = (a2 - lambda2) - (a1 - lambda1)
F1_pi = gamma * phi * m1 * pi1 + 0.5 * gamma**2 * phi**2 * sigma1**2 * pi1**2
F1_p = gamma * phi * lambda1 * p1 + 0.5 * gamma**2 * phi**2 * b1**2 * p1**2
F2_pi = -gamma * phi * m2 * pi2 + 0.5 * gamma**2 * phi**2 * sigma2**2 * pi2**2
F2_p = -gamma * phi * lambda2 * p2 + 0.5 * gamma**2 * phi**2 * b2**2 * p2**2
integrand = F1_pi + F1_p + F2_pi + F2_p - gamma * D * phi
areas = 0.5 * (integrand[:-1] + integrand[1:]) * np.diff(t)
C = np.zeros_like(t)
C[:-1] = np.cumsum(areas[::-1])[::-1]

# Distribution of the equilibrium terminal wealth difference at t=0:
# M(0,x)=exp(rT)x+mean_constant and Q(0)=terminal_variance.
equilibrium_drift = m2 * pi2 - m1 * pi1 + D + lambda2 * p2 - lambda1 * p1
equilibrium_variance_rate = (
    sigma1**2 * pi1**2 + sigma2**2 * pi2**2
    + b1**2 * p1**2 + b2**2 * p2**2
)
mean_constant = trapezoid(phi * equilibrium_drift, t)
terminal_variance = trapezoid(phi**2 * equilibrium_variance_rate, t)
certainty_equivalent_constant = mean_constant - 0.5 * gamma * terminal_variance
initial_gap_ce_threshold = -certainty_equivalent_constant / np.exp(r * T)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "legend.fontsize": 7.3,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

blue = "#1f5a94"
orange = "#c25b23"
grey = "#666666"

fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)

ax = axes[0]
ax.plot(t, pi1, color=orange, lw=1.8, label=r"$\pi_1^*(t)$")
ax.plot(t, pi2, color=blue, lw=1.8, label=r"$\pi_2^*(t)$")
ax.plot(t, pi2_hat, color=blue, lw=1.0, ls="--", alpha=0.75,
        label=r"$\widehat\pi_2(t)$")
ax.axvline(t_pi, color=grey, lw=0.9, ls=":")
ax.annotate(
    rf"$t_\pi={t_pi:.2f}$",
    xy=(t_pi, pi2_bar),
    xytext=(t_pi - 2.5, pi2_bar - 0.18),
    arrowprops={"arrowstyle": "->", "lw": 0.7, "color": grey},
)
ax.set(xlabel=r"Time $t$", ylabel="Amount invested", title="(a) Investment controls")
ax.set_xlim(0, T)
ax.set_ylim(1.35, 2.3)
ax.grid(alpha=0.22, lw=0.5)
ax.legend(frameon=False, loc="lower right")

ax = axes[1]
ax.plot(t, p1, color=orange, lw=1.8, label=r"$p_1^*(t)$")
ax.plot(t, p2, color=blue, lw=1.8, label=r"$p_2^*(t)$")
ax.plot(t, p2_hat, color=blue, lw=1.0, ls="--", alpha=0.75,
        label=r"$\widehat p_2(t)$")
ax.axvline(t_p, color=grey, lw=0.9, ls=":")
ax.annotate(
    rf"$t_p={t_p:.2f}$",
    xy=(t_p, 1.0),
    xytext=(t_p - 2.6, 0.91),
    arrowprops={"arrowstyle": "->", "lw": 0.7, "color": grey},
)
ax.set(xlabel=r"Time $t$", ylabel="Retention level", title="(b) Reinsurance controls")
ax.set_xlim(0, T)
ax.set_ylim(0.78, 1.18)
ax.grid(alpha=0.22, lw=0.5)
ax.legend(frameon=False, loc="lower right")

fig.savefig(FIG_DIR / "baseline_controls.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "baseline_controls.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)
ax.plot(t, C, color=blue, lw=1.8)
ax.axvline(t_pi, color=grey, lw=0.9, ls=":")
ax.axvline(t_p, color=grey, lw=0.9, ls=":")
ax.scatter([t_pi, t_p], np.interp([t_pi, t_p], t, C), color=orange, s=18, zorder=3)
ax.annotate(r"$t_\pi$", xy=(t_pi, np.interp(t_pi, t, C)),
            xytext=(t_pi - 1.2, np.interp(t_pi, t, C) + 0.25),
            arrowprops={"arrowstyle": "->", "lw": 0.7, "color": grey})
ax.annotate(r"$t_p$", xy=(t_p, np.interp(t_p, t, C)),
            xytext=(t_p + 0.6, np.interp(t_p, t, C) + 0.2),
            arrowprops={"arrowstyle": "->", "lw": 0.7, "color": grey})
ax.set(xlabel=r"Time $t$", ylabel=r"Coefficient $C(t)$")
ax.set_xlim(0, T)
ax.grid(alpha=0.22, lw=0.5)
fig.savefig(FIG_DIR / "value_coefficient.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "value_coefficient.png", dpi=300, bbox_inches="tight")
plt.close(fig)


relative_change = np.array([-30.0, -15.0, 0.0, 15.0, 30.0])
factor = 1.0 + relative_change / 100.0


def tau_pi(g=gamma, m=m2, sig=sigma2, cap=pi2_bar):
    a = m / (g * sig**2)
    return clipped_switch_time(a, cap)


def tau_p(g=gamma, loading=lambda2, under_vol=b2):
    b = loading / (g * under_vol**2)
    return clipped_switch_time(b, 1.0)


pi_sensitivity = {
    r"Risk aversion $\gamma$": [tau_pi(g=gamma * q) for q in factor],
    r"Excess return $\mu_2-r$": [tau_pi(m=m2 * q) for q in factor],
    r"Market volatility $\sigma_2$": [tau_pi(sig=sigma2 * q) for q in factor],
    r"Investment cap $\overline{\pi}_2$": [tau_pi(cap=pi2_bar * q) for q in factor],
}
p_sensitivity = {
    r"Risk aversion $\gamma$": [tau_p(g=gamma * q) for q in factor],
    r"Reinsurance price $\lambda_2$": [tau_p(loading=lambda2 * q) for q in factor],
    r"Underwriting volatility $b_2$": [tau_p(under_vol=b2 * q) for q in factor],
}

markers = ["o", "s", "^", "D"]
fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.95), constrained_layout=True)

ax = axes[0]
for (label, values), marker in zip(pi_sensitivity.items(), markers):
    ax.plot(relative_change, values, marker=marker, ms=3.2, lw=1.25, label=label)
ax.axhline(t_pi, color=grey, lw=0.7, ls=":")
ax.set(
    xlabel="Change from baseline (%)",
    ylabel="Boundary-entry time",
    title=r"(a) Investment entry time $\tau_\pi$",
)
ax.set_xlim(-30, 30)
ax.set_ylim(-0.15, T + 0.15)
ax.set_yticks(np.arange(0, T + 1, 2))
ax.grid(alpha=0.22, lw=0.5)
ax.legend(frameon=False, loc="center right")

ax = axes[1]
for (label, values), marker in zip(p_sensitivity.items(), markers):
    ax.plot(relative_change, values, marker=marker, ms=3.2, lw=1.25, label=label)
ax.axhline(t_p, color=grey, lw=0.7, ls=":")
ax.set(
    xlabel="Change from baseline (%)",
    ylabel="Boundary-entry time",
    title=r"(b) Retention entry time $\tau_p$",
)
ax.set_xlim(-30, 30)
ax.set_ylim(-0.15, T + 0.15)
ax.set_yticks(np.arange(0, T + 1, 2))
ax.grid(alpha=0.22, lw=0.5)
ax.legend(frameon=False, loc="center right")

fig.savefig(FIG_DIR / "switching_sensitivity.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "switching_sensitivity.png", dpi=300, bbox_inches="tight")
plt.close(fig)

for filename, title, series_data, baseline_time in [
    ("investment_switch_sensitivity", r"Investment entry time $\tau_\pi$",
     pi_sensitivity, t_pi),
    ("retention_switch_sensitivity", r"Retention entry time $\tau_p$",
     p_sensitivity, t_p),
]:
    fig, ax = plt.subplots(figsize=(3.35, 2.55), constrained_layout=True)
    for (label, values), marker in zip(series_data.items(), markers):
        ax.plot(relative_change, values, marker=marker, ms=3.0, lw=1.15,
                label=label)
    ax.axhline(baseline_time, color=grey, lw=0.7, ls=":")
    ax.set(
        xlabel="Change from baseline (%)",
        ylabel="Boundary-entry time",
        title=title,
    )
    ax.set_xlim(-30, 30)
    ax.set_ylim(-0.15, T + 0.15)
    ax.set_yticks(np.arange(0, T + 1, 2))
    ax.grid(alpha=0.22, lw=0.5)
    ax.legend(frameon=False, loc="center right", fontsize=6.4)
    fig.savefig(FIG_DIR / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

print(f"A2={A2:.8f}")
print(f"B2={B2:.8f}")
print(f"pi2(0)={pi2[0]:.8f}")
print(f"p2(0)={p2[0]:.8f}")
print(f"t_pi={t_pi:.8f}")
print(f"t_p={t_p:.8f}")
print(f"D={D:.8f}")
print(f"C(0)={C[0]:.8f}")
print(f"C(t_pi)={np.interp(t_pi, t, C):.8f}")
print(f"C(t_p)={np.interp(t_p, t, C):.8f}")
print(f"M(0,x)=exp(rT)x+{mean_constant:.8f}")
print(f"Q(0)={terminal_variance:.8f}")
print(f"sd(0)={np.sqrt(terminal_variance):.8f}")
print(f"CE(0,x)=exp(rT)x{certainty_equivalent_constant:+.8f}")
print(f"CE initial-gap threshold={initial_gap_ce_threshold:.8f}")
print("pi sensitivity")
for key, values in pi_sensitivity.items():
    print(key, [round(v, 4) for v in values])
print("retention sensitivity")
for key, values in p_sensitivity.items():
    print(key, [round(v, 4) for v in values])


# Six company-by-company sensitivity figures.  For company-specific quantities,
# the same percentage change is applied to each company's own parameter.  For
# example, the excess-return experiment varies m_1 and m_2 proportionally; it
# does not insert a Company 2 parameter into Company 1's control.
sensitivity_specs = [
    {
        "key": "gamma",
        "filename": "sensitivity_cara_parameter",
        "caption": r"Risk aversion $\gamma$",
        "affected": {"c2_pi", "c2_p"},
    },
    {
        "key": "excess_return",
        "caption": r"Excess return $\mu_i-r$",
        "affected": {"c2_pi"},
    },
    {
        "key": "market_volatility",
        "caption": r"Market volatility $\sigma_i$",
        "affected": {"c2_pi"},
    },
    {
        "key": "investment_cap",
        "caption": r"Investment cap $\overline{\pi}_i$",
        "affected": {"c1_pi", "c2_pi"},
    },
    {
        "key": "reinsurance_price",
        "caption": r"Reinsurance price $\lambda_i$",
        "affected": {"c2_p"},
    },
    {
        "key": "underwriting_volatility",
        "caption": r"Underwriting volatility $b_i$",
        "affected": {"c2_p"},
    },
]


def strategies_for_change(key, q):
    """Return equilibrium paths after one proportional parameter change."""
    g = gamma * q if key == "gamma" else gamma
    m1_q = m1 * q if key == "excess_return" else m1
    m2_q = m2 * q if key == "excess_return" else m2
    sig1_q = sigma1 * q if key == "market_volatility" else sigma1
    sig2_q = sigma2 * q if key == "market_volatility" else sigma2
    cap1_q = pi1_bar * q if key == "investment_cap" else pi1_bar
    cap2_q = pi2_bar * q if key == "investment_cap" else pi2_bar
    lam1_q = lambda1 * q if key == "reinsurance_price" else lambda1
    lam2_q = lambda2 * q if key == "reinsurance_price" else lambda2
    b1_q = b1 * q if key == "underwriting_volatility" else b1
    b2_q = b2 * q if key == "underwriting_volatility" else b2

    # Under K_1=[0,bar pi_1], m_1>0 and lambda_1>0 throughout the
    # experiment, Company 1 selects the upper endpoints.
    c1_pi = np.full_like(t, cap1_q)
    c1_p = np.ones_like(t)

    c2_pi_hat = m2_q / (g * sig2_q**2) * discount
    c2_p_hat = lam2_q / (g * b2_q**2) * discount
    c2_pi = np.minimum(cap2_q, c2_pi_hat)
    c2_p = np.minimum(1.0, c2_p_hat)

    # Keep the unused Company 1 parameters explicit: they affect the payoff
    # and endpoint comparison but not the selected endpoint in this baseline.
    _ = (m1_q, sig1_q, lam1_q, b1_q)
    return c1_pi, c2_pi, c1_p, c2_p


scenario_colors = mpl.colormaps["viridis"](np.linspace(0.08, 0.92, len(factor)))
scenario_labels = [f"{v:+.0f}%" if v else "Baseline" for v in relative_change]

for spec in sensitivity_specs:
    paths = [strategies_for_change(spec["key"], q) for q in factor]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 3.85), constrained_layout=True)
    panels = [
        (axes[0, 0], 0, "c1_pi", "Company 1: investment", "Amount invested"),
        (axes[0, 1], 1, "c2_pi", "Company 2: investment", "Amount invested"),
        (axes[1, 0], 2, "c1_p", "Company 1: retention", "Retention level"),
        (axes[1, 1], 3, "c2_p", "Company 2: retention", "Retention level"),
    ]

    for ax, path_index, panel_key, title, ylabel in panels:
        if panel_key in spec["affected"]:
            for idx, (values, color, label) in enumerate(
                zip([p[path_index] for p in paths], scenario_colors, scenario_labels)
            ):
                is_baseline = relative_change[idx] == 0
                ax.plot(
                    t,
                    values,
                    color="black" if is_baseline else color,
                    lw=1.8 if is_baseline else 1.15,
                    ls="--" if is_baseline else "-",
                    label=label,
                )
        else:
            ax.plot(t, paths[2][path_index], color=grey, lw=1.8,
                    label="All changes coincide")
            ax.text(0.04, 0.11, "No direct control effect",
                    transform=ax.transAxes, color=grey, fontsize=7.2)

        ax.set(xlabel=r"Time $t$", ylabel=ylabel, title=title)
        ax.set_xlim(0, T)
        if path_index in (2, 3):
            panel_min = min(float(np.min(p[path_index])) for p in paths)
            ax.set_ylim(max(0.0, panel_min - 0.05), 1.05)
        elif spec["key"] == "investment_cap" or path_index == 1:
            ax.set_ylim(0.95, 2.95)
        else:
            ax.set_ylim(1.25, 1.72)
        ax.grid(alpha=0.22, lw=0.5)
        ax.legend(frameon=False, fontsize=6.5, loc="best", ncol=2)

    fig.suptitle(spec["caption"], fontsize=10.2)
    figure_stem = spec.get("filename", f"sensitivity_{spec['key']}")
    fig.savefig(FIG_DIR / f"{figure_stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{figure_stem}.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


print("selected endpoint switching times (-30%, baseline, +30%)")
for q, label in [(0.70, "-30%"), (1.0, "baseline"), (1.30, "+30%")]:
    print(
        label,
        {
            "gamma": (tau_pi(g=gamma*q), tau_p(g=gamma*q)),
            "m2": tau_pi(m=m2*q),
            "sigma2": tau_pi(sig=sigma2*q),
            "pi2_bar": tau_pi(cap=pi2_bar*q),
            "lambda2": tau_p(loading=lambda2*q),
            "b2": tau_p(under_vol=b2*q),
        },
    )
