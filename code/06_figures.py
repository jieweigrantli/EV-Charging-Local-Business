"""
06_figures.py
-------------
Draws every main and supplementary figure from the per-panel CSV files written
by scripts 01-05 (outputs/source_data).  Each figure is saved as a vector PDF
and a 300-dpi PNG in outputs/figures.

Main figures
  Figure_1  Visit frequency and spending magnitude by amenity type (a: bars; b: box + dots)
  Figure_2  Amenity counts within 250 m of charging locations (discrete distributions, batlow colours)
  Figure_3  Predicted probability of spending for significant covariates (six panels)
  Figure_4  Visit share (bars) and transaction amounts (boxplots) by reason for charging (two panels)
Supplementary figures
  Supplementary_Figure_1/2  Monte Carlo convergence (stage 1 / stage 2)
  Supplementary_Figure_5    Counterfactual charge-duration scenarios
  Supplementary_Figure_6    Demographic benchmarks
(Supplementary Figure 3, the map, is drawn by 07_map.py; Supplementary Figure 4
is a static illustration produced with the Google Places API.)
"""
import json

import cmcrameri.cm as cmc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import (FIG_DIR, PALETTE_PRIMARY, PALETTE_QUATERNARY, PALETTE_SECONDARY, PALETTE_TERTIARY,
                    REASON_COLORS, REASON_ORDER, ROOT, SD_DIR, STORE_TYPES)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
# Optional seventh panel for Figure 3 (standalone restrooms, P = 0.040).
FIG3_INCLUDE_STANDALONE_RESTROOM = False


def panel_label(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig]  outputs/figures/{name}.pdf (+png)")


# =========================================================================== #
# Figure 1
# =========================================================================== #
def figure_1():
    visits = pd.read_csv(SD_DIR / "fig1a_visits_by_store.csv").sort_values("visits")
    tx = pd.read_csv(SD_DIR / "fig1b_transactions_by_store.csv")
    order = visits["store_type"].tolist()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(order))
    ax1.barh(y, visits["visits"], color=PALETTE_QUATERNARY, alpha=0.85)
    for yi, v in zip(y, visits["visits"]):
        ax1.text(v + 4, yi, f"{v}", va="center", fontsize=7.5, color="0.3")
    ax1.set_yticks(y)
    ax1.set_yticklabels(order)
    ax1.set_xlabel("Number of reported transactions ($n$)")
    ax1.set_xlim(0, visits["visits"].max() * 1.12)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    panel_label(ax1, "a", x=-0.55)

    rng = np.random.default_rng(0)
    data = [np.log1p(tx.loc[tx["store_type"] == s, "amount"].values) for s in order]
    bp = ax2.boxplot(data, positions=y, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                     whis=1.5, boxprops=dict(facecolor="none", edgecolor=PALETTE_PRIMARY, linewidth=1.3),
                     medianprops=dict(color=PALETTE_PRIMARY, linewidth=1.8),
                     whiskerprops=dict(color=PALETTE_PRIMARY, linewidth=1.1),
                     capprops=dict(color=PALETTE_PRIMARY, linewidth=1.1), zorder=2)
    for yi, d in zip(y, data):
        jitter = rng.uniform(-0.24, 0.24, size=len(d))
        ax2.scatter(d, yi + jitter, s=9, color=PALETTE_SECONDARY, alpha=0.5, linewidths=0, zorder=3)
        ax2.plot(np.log1p(np.expm1(d).mean()), yi, marker="D", color="black", markersize=4.5, zorder=4)
    ticks = [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000]
    ax2.set_xticks(np.log1p(ticks))
    ax2.set_xticklabels([f"{t:,}" for t in ticks])
    ax2.set_xlabel("Amount spent per transaction (US$, log scale)")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.set_yticks(y)
    ax1.set_yticklabels(order)
    ax2.legend(handles=[
        Line2D([0], [0], marker="o", color=PALETTE_SECONDARY, lw=0, markersize=4, alpha=0.6, label="Individual transaction"),
        Patch(facecolor="none", edgecolor=PALETTE_PRIMARY, label="Median and interquartile range"),
        Line2D([0], [0], marker="D", color="black", lw=0, markersize=4, label="Mean")],
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=7.5)
    panel_label(ax2, "b", x=-0.04)
    save(fig, "Figure_1")


# =========================================================================== #
# Figure 2: discrete distribution of amenity counts (batlow colours)
# =========================================================================== #
def figure_2():
    dist = pd.read_csv(SD_DIR / "fig2_amenity_count_distribution.csv")
    stats = pd.read_csv(SD_DIR / "fig2_amenity_count_summary.csv")
    order = stats["amenity"].tolist()
    n_loc = int(stats["n"].iloc[0])
    labels = {"restaurant": "Restaurant", "retail": "Retail store", "department_store": "Department store",
              "grocery": "Grocery store", "work": "Workplace", "medical": "Medical facility",
              "colocated_bathroom": "Co-located restroom", "park": "Park", "religion": "Religious facility",
              "entertainment": "Entertainment", "standalone_bathroom": "Standalone restroom"}
    colors = cmc.batlow(np.linspace(0.05, 0.9, len(order)))
    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.9 * nrow))
    axes = axes.ravel()
    st = stats.set_index("amenity")
    for ax, a, col in zip(axes, order, colors):
        d = dist[dist["amenity"] == a].sort_values("count")
        full = pd.Series(0, index=np.arange(0, int(d["count"].max()) + 1))
        full.loc[d["count"].values] = d["n_locations"].values
        ax.bar(full.index, full.values, width=0.85, color=col, edgecolor="none", zorder=2)
        q1, med, q3 = st.loc[a, "q1"], st.loc[a, "median"], st.loc[a, "q3"]
        ax.axvspan(q1 - 0.5, q3 + 0.5, color="0.5", alpha=0.18, lw=0, zorder=1)
        ax.axvline(med, color="black", lw=1.4, ls="--", zorder=3)
        ax.set_title(labels[a], fontsize=10, loc="left", pad=4)
        ax.text(0.98, 0.95, f"median {med:g}\nIQR {q1:g}\u2013{q3:g}", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="0.25")
        ax.set_xlim(-0.7, max(full.index.max() + 0.7, 5))
        ax.set_ylabel("Locations")
        ax.set_xlabel("Amenities within 250 m (count)")
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        ax.margins(y=0.08)
    for ax in axes[len(order):]:
        ax.axis("off")
    axes[len(order)].legend(handles=[Line2D([0], [0], color="black", lw=1.4, ls="--", label="Median"),
                                     Patch(facecolor="0.5", alpha=0.18, label="Interquartile range")],
                            loc="center", frameon=False, fontsize=9, title=f"$n$ = {n_loc:,} charging locations",
                            title_fontsize=9)
    fig.tight_layout(h_pad=1.8, w_pad=1.5)
    save(fig, "Figure_2")


# =========================================================================== #
# Figure 3: predicted probabilities (six panels, R-figure style)
# =========================================================================== #
def figure_3():
    specs = json.load(open(SD_DIR / "fig3_panels.json"))
    order = ["battery_capacity_centered", "charge_duration", "hh_children", "250m_medical",
             "charge_session_reason_long_trip", "charge_session_reason_opportunity"]
    if FIG3_INCLUDE_STANDALONE_RESTROOM:
        order.append("250m_standalone_bathroom")
    specs = {s["var"]: s for s in specs}
    order = [v for v in order if v in specs]
    nrow = int(np.ceil(len(order) / 2))
    fig, axes = plt.subplots(nrow, 2, figsize=(9.2, 3.6 * nrow))
    axes = axes.ravel()
    n = specs[order[0]]["n"]
    for ax, var, letter in zip(axes, order, "abcdefg"):
        s = specs[var]
        safe = var.replace("250m_", "X250m_")
        m = pd.read_csv(SD_DIR / f"fig3_marginal_{safe}.csv")
        d = pd.read_csv(SD_DIR / f"fig3_distribution_{safe}.csv")
        ax2 = ax.twinx()
        ax2.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)
        if s["kind"] == "continuous":
            ax2.bar((d["bin_left"] + d["bin_right"]) / 2, d["n_sessions"], width=(d["bin_right"] - d["bin_left"]),
                    color="0.82", alpha=0.7, edgecolor="none")
            ax.plot(m["x"], m["predicted_probability"], color=PALETTE_PRIMARY, lw=1.6, zorder=5)
            ax.fill_between(m["x"], m["ci_low_95"], m["ci_high_95"], color=PALETTE_PRIMARY, alpha=0.18, lw=0, zorder=4)
        else:
            ax2.bar(d["x"], d["n_sessions"], width=0.9, color="0.82", alpha=0.7, edgecolor="none")
            ax.errorbar(m["x"], m["predicted_probability"],
                        yerr=[m["predicted_probability"] - m["ci_low_95"], m["ci_high_95"] - m["predicted_probability"]],
                        fmt="o", color=PALETTE_PRIMARY, markersize=3.5, capsize=3, lw=1.1, zorder=5)
            if s["kind"] == "binary":
                ax.set_xticks([0, 1]); ax.set_xticklabels(["No", "Yes"]); ax.set_xlim(-0.7, 1.7)
            else:
                ax.set_xticks(m["x"][::2] if len(m) > 10 else m["x"])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Probability of spending", color=PALETTE_PRIMARY)
        ax.tick_params(axis="y", colors=PALETTE_PRIMARY)
        ax.set_xlabel(s["xlabel"])
        ax2.set_ylabel("Sessions (count)", color="0.4")
        ax2.tick_params(axis="y", colors="0.4")
        ax2.spines["right"].set_visible(True)
        ax2.spines["top"].set_visible(False)
        ax.grid(True, linestyle="--", lw=0.5, alpha=0.5)
        panel_label(ax, letter, x=-0.16, y=1.02)
    for ax in axes[len(order):]:
        ax.axis("off")
    fig.tight_layout(h_pad=2.0, w_pad=3.0)
    save(fig, "Figure_3")
    return n


# =========================================================================== #
# Figure 4: heterogeneity by reason for charging (two panels)
# =========================================================================== #
def figure_4():
    a = pd.read_csv(SD_DIR / "fig4a_visit_share_by_reason.csv")
    b = pd.read_csv(SD_DIR / "fig4b_transactions_by_reason.csv")
    stores = list(STORE_TYPES.values())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7.2), gridspec_kw={"wspace": 0.18})
    x = np.arange(len(stores))
    w = 0.26
    n_resp = {r: int(a.loc[a["reason"] == r, "n_respondents"].iloc[0]) for r in REASON_ORDER}
    for j, r in enumerate(REASON_ORDER):
        sub = a[a["reason"] == r].set_index("store_type").reindex(stores)
        ax1.bar(x + (j - 1) * w, sub["percent"], width=w, color=REASON_COLORS[r], yerr=sub["se_percent"],
                capsize=3, error_kw={"elinewidth": 0.9, "ecolor": "black"},
                label=f"{r} ($n$ = {n_resp[r]})")
    counts_a = {s: [int(a[(a["reason"] == r) & (a["store_type"] == s)]["visits"].iloc[0]) for r in REASON_ORDER] for s in stores}
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s}\n(O: {c[0]}, L: {c[1]}, R: {c[2]})" for s, c in counts_a.items()], rotation=45, ha="right")
    ax1.set_ylabel("Respondents visiting the amenity type (%)")
    ax1.legend(title="Reason for charging (respondents)", loc="upper right", frameon=False)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    panel_label(ax1, "a", x=-0.08)

    n_tx = {r: int((b["reason"] == r).sum()) for r in REASON_ORDER}
    counts_b = {s: [] for s in stores}
    for j, r in enumerate(REASON_ORDER):
        data = []
        for s in stores:
            vals = b.loc[(b["reason"] == r) & (b["store_type"] == s), "amount"].values
            counts_b[s].append(len(vals))
            data.append(vals if len(vals) else np.array([np.nan]))
        ax2.boxplot(data, positions=x + (j - 1) * w, widths=w * 0.85, patch_artist=True, whis=1.5,
                    boxprops=dict(facecolor=REASON_COLORS[r], edgecolor="black", linewidth=0.8),
                    medianprops=dict(color="black", linewidth=1.3), whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="black",
                                    markersize=4, markeredgewidth=0.7),
                    showmeans=True, meanprops=dict(marker="^", markerfacecolor="none", markeredgecolor=PALETTE_PRIMARY,
                                                   markersize=6, markeredgewidth=1.3))
    ax2.set_ylabel("Amount spent per transaction (US$)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{s}\n(O: {c[0]}, L: {c[1]}, R: {c[2]})" for s, c in counts_b.items()], rotation=45, ha="right")
    ax2.legend(handles=[Patch(facecolor=REASON_COLORS[r], edgecolor="black", label=f"{r} ($n$ = {n_tx[r]})") for r in REASON_ORDER]
               + [Line2D([0], [0], marker="^", color=PALETTE_PRIMARY, markerfacecolor="none", lw=0, markersize=6, label="Mean"),
                  Line2D([0], [0], marker="o", color="black", markerfacecolor="none", markeredgewidth=0.7, lw=0, markersize=4,
                         label="Outlier")],
               title="Reason for charging (transactions)", loc="upper right", frameon=False)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    panel_label(ax2, "b", x=-0.08)
    save(fig, "Figure_4")


# =========================================================================== #
# Supplementary figures
# =========================================================================== #
def supp_fig_convergence():
    s1 = pd.read_csv(SD_DIR / "suppfig1_stage1_convergence.csv")
    s2 = pd.read_csv(SD_DIR / "suppfig2_stage2_convergence.csv")
    for name, s, cols, labels, vline in (
        ("Supplementary_Figure_1", s1, ["mean_mcfadden_pseudo_r2", "mean_coef_charge_duration", "mean_coef_long_trip"],
         ["McFadden's pseudo-$R^2$", "Charge-duration coefficient (log-odds)", "Long-trip coefficient (log-odds)"], 1000),
        ("Supplementary_Figure_2", s2, ["mean_adj_r2", "mean_coef_charge_duration", "mean_coef_battery_capacity"],
         ["Adjusted $R^2$", "Charge-duration coefficient", "Battery-capacity coefficient"], 400),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, c, lab, col, letter in zip(axes, cols, labels, ["black", PALETTE_PRIMARY, "#7E3F97"], "abc"):
            ax.plot(s["sample_size"], s[c], marker="o", markersize=4, lw=1.6, color=col)
            ax.axvline(vline, color="darkgreen", linestyle="--", lw=1.2)
            ax.set_xlabel("Sub-sample size ($N$)")
            ax.set_ylabel(f"Mean {lab}" if not lab.startswith("Mc") and not lab.startswith("Adj") else lab)
            ax.grid(True, linestyle="--", alpha=0.4)
            panel_label(ax, letter, x=-0.22)
        fig.tight_layout(w_pad=2.5)
        save(fig, name)


def supp_fig_counterfactual():
    cf = pd.read_csv(SD_DIR / "suppfig5_counterfactual.csv")
    d = cf[cf["scenario_type"] == "delta_minutes"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, c, lab, col, letter in zip(axes,
            ["pct_change_probability", "pct_change_expected_spend", "abs_change_expected_spend_usd"],
            ["Change in probability of spending (%)", "Change in expected ancillary spend (%)",
             "Change in expected ancillary spend (US$ per session)"],
            [PALETTE_PRIMARY, PALETTE_SECONDARY, PALETTE_TERTIARY], "abc"):
        ax.plot(d["delta_minutes"], d[c], marker="o", markersize=5, lw=1.8, color=col)
        ax.axhline(0, color="0.4", linestyle="--", lw=1)
        ax.axvline(0, color="0.6", linestyle=":", lw=1)
        ax.set_xlabel("Shift in charge duration (minutes)")
        ax.set_ylabel(lab)
        ax.grid(True, linestyle="--", alpha=0.4)
        panel_label(ax, letter, x=-0.22)
    fig.tight_layout(w_pad=2.5)
    save(fig, "Supplementary_Figure_5")


def supp_fig_demographics():
    demo = pd.read_csv(SD_DIR / "suppfig6_demographic_comparison.csv")
    orders = {"Household income": ["<$50k", "$50k-$99k", "$100k-$149k", "$150k-$199k", "$200k+"],
              "Education": ["<High school", "High school/GED", "Some college/Associate", "Bachelor's", "Graduate/professional"],
              "Age": ["18-34", "35-44", "45-54", "55-64", "65-74", "75+"]}
    sources = ["Survey sample", "California EV owners", "National EV owners"]
    colors = {"Survey sample": PALETTE_PRIMARY, "California EV owners": PALETTE_TERTIARY, "National EV owners": PALETTE_SECONDARY}
    metrics = [m for m in orders if m in demo["Metric"].unique()]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 4.2))
    axes = np.atleast_1d(axes)
    for ax, metric, letter in zip(axes, metrics, "abc"):
        cats = orders[metric]
        x = np.arange(len(cats)); w = 0.27
        for j, src in enumerate(sources):
            sub = demo[(demo["Metric"] == metric) & (demo["Source"] == src)].set_index("Category").reindex(cats)
            ax.bar(x + (j - 1) * w, sub["Percent"], width=w, color=colors[src], label=src)
        ax.set_xticks(x); ax.set_xticklabels([c.replace("$", "\\$") for c in cats], rotation=35, ha="right")
        ax.set_ylabel("Share of respondents (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        panel_label(ax, letter, x=-0.2)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=2.0)
    save(fig, "Supplementary_Figure_6")


if __name__ == "__main__":
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    supp_fig_convergence()
    supp_fig_counterfactual()
    supp_fig_demographics()
    print("[done] 06_figures.py")
