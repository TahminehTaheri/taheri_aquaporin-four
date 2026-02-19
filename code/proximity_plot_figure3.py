## run scripts: proximity_SC and proximity_eu before running this file
## this code needs some generated data from above mentioned codes

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_npz(npz_path):
    dat = np.load(npz_path, allow_pickle=True)

    ph = dat["phenotypes"].astype(object)
    obs = np.asarray(dat["obs_means"], float)
    null = np.asarray(dat["null_means"], float)
    p_raw = np.asarray(dat["p_raw"], float)
    q_fdr = np.asarray(dat["q_fdr"], float)

    obs_aqp1 = np.asarray(dat["obs_means_aqp1"], float) if "obs_means_aqp1" in dat else None
    obs_aqp9 = np.asarray(dat["obs_means_aqp9"], float) if "obs_means_aqp9" in dat else None

    idx = {str(p): i for i, p in enumerate(ph)}
    return idx, ph, obs, null, p_raw, q_fdr, obs_aqp1, obs_aqp9

# paths
root = Path("/Users/tahminehtaheri/aquaporin_four")
res_dir = root / "results"
fig_dir = root / "figures" / "proximity"
fig_dir.mkdir(parents=True, exist_ok=True)

eu_fp = res_dir / "euclid_proximity_moran_boxplot_data.npz"
sc_fp = res_dir / "sc_proximity_moran_boxplot_data.npz"

out_box = fig_dir / "combined_proximity_EU_vs_SC_ordered.svg"
out_lolli = fig_dir / "lollipop_proximity_AQP4_vs_AQP1_vs_AQP9_EU_vs_SC_ordered.svg"

# desired order for boxplots
order_top_to_bottom = [
    "Edema",
    "TDP43C",
    "TDP43A",
    "PS1",
    "LOA",
    "EOA",
    "DLB",
    "4Rtau",
    "3Rtau",
]

# desired order for lollipops 
order_left_to_right = ["EOA", "PS1", "3Rtau", "4Rtau", "TDP43A", "TDP43C", "DLB", "LOA", "Edema"]

display_name = {
    "LOA": "LOAD",
    "EOA": "EOAD",
    "Edema": "edema",
    "TDP43C": "TDP-43C",
    "TDP43A": "TDP-43A",
    "PS1": "PS1",
    "DLB": "DLB",
    "4Rtau": "4Rtau",
    "3Rtau": "3Rtau",
}

eu_idx, eu_ph, eu_obs, eu_null, eu_p, eu_q, eu_obs_aqp1_all, eu_obs_aqp9_all = load_npz(eu_fp)
sc_idx, sc_ph, sc_obs, sc_null, sc_p, sc_q, sc_obs_aqp1_all, sc_obs_aqp9_all = load_npz(sc_fp)

# colors
blue = "#2b8cbe"
red = "#de2d26"
grey = "#6b7280"

col_eu = blue
col_sc = red

# boxplots
order_box = [p for p in order_top_to_bottom if (p in eu_idx) and (p in sc_idx)]
if len(order_box) == 0:
    raise RuntimeError("No overlapping phenotypes between EU and SC npz files.")

eu_null_list = [eu_null[eu_idx[p]][np.isfinite(eu_null[eu_idx[p]])] for p in order_box]
sc_null_list = [sc_null[sc_idx[p]][np.isfinite(sc_null[sc_idx[p]])] for p in order_box]

eu_obs_list = [float(eu_obs[eu_idx[p]]) for p in order_box]
sc_obs_list = [float(sc_obs[sc_idx[p]]) for p in order_box]

eu_p_list = [float(eu_p[eu_idx[p]]) for p in order_box]
sc_p_list = [float(sc_p[sc_idx[p]]) for p in order_box]

eu_q_list = [float(eu_q[eu_idx[p]]) for p in order_box]
sc_q_list = [float(sc_q[sc_idx[p]]) for p in order_box]

labels_y = [display_name.get(p, p) for p in order_box]
disease_set = set([p for p in order_box if p != "Edema"])

n = len(order_box)
pos = np.arange(1, n + 1)
delta = 0.18
pos_eu = pos + delta
pos_sc = pos - delta

fig, ax = plt.subplots(figsize=(9, 1 + 0.9 * n), dpi=300)

bp_eu = ax.boxplot(
    eu_null_list, positions=pos_eu, vert=False, patch_artist=True,
    widths=0.30, showfliers=False, labels=None,
    medianprops=dict(color=col_eu, linewidth=2)
)
for box in bp_eu["boxes"]:
    box.set(facecolor=col_eu, edgecolor=col_eu, alpha=0.25)
for k in ["whiskers", "caps"]:
    for line in bp_eu[k]:
        line.set(color=col_eu, linewidth=1.5)

bp_sc = ax.boxplot(
    sc_null_list, positions=pos_sc, vert=False, patch_artist=True,
    widths=0.30, showfliers=False, labels=None,
    medianprops=dict(color=col_sc, linewidth=2)
)
for box in bp_sc["boxes"]:
    box.set(facecolor=col_sc, edgecolor=col_sc, alpha=0.25)
for k in ["whiskers", "caps"]:
    for line in bp_sc[k]:
        line.set(color=col_sc, linewidth=1.5)

ax.scatter(eu_obs_list, pos_eu, s=55, color=col_eu, edgecolor="white", linewidth=0.9, zorder=3)
ax.scatter(sc_obs_list, pos_sc, s=55, color=col_sc, edgecolor="white", linewidth=0.9, zorder=3)

ax.set_yticks(pos)
ax.set_yticklabels(labels_y)
ax.invert_yaxis()

max_x = max(
    np.nanmax([np.nanmax(v) for v in eu_null_list]),
    np.nanmax([np.nanmax(v) for v in sc_null_list]),
    np.nanmax(eu_obs_list),
    np.nanmax(sc_obs_list),
)
ax.set_xlim(0, max_x * 1.22)

x_txt = ax.get_xlim()[1] * 0.985

for y0, ph, p0, q0 in zip(pos_eu, order_box, eu_p_list, eu_q_list):
    if ph == "Edema":
        star = "*" if p0 < 0.05 else ""
        ax.text(x_txt, y0, f"$p$={p0:.3g}{star}", va="center", ha="left", fontsize=10, color=col_eu)
    else:
        star = "*" if (np.isfinite(q0) and q0 < 0.05) else ""
        ax.text(x_txt, y0, f"$q_\\mathrm{{FDR}}$={q0:.3g}{star}", va="center", ha="left", fontsize=10, color=col_eu)

for y0, ph, p0, q0 in zip(pos_sc, order_box, sc_p_list, sc_q_list):
    if ph == "Edema":
        star = "*" if p0 < 0.05 else ""
        ax.text(x_txt, y0, f"$p$={p0:.3g}{star}", va="center", ha="left", fontsize=10, color=col_sc)
    else:
        star = "*" if (np.isfinite(q0) and q0 < 0.05) else ""
        ax.text(x_txt, y0, f"$q_\\mathrm{{FDR}}$={q0:.3g}{star}", va="center", ha="left", fontsize=10, color=col_sc)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("Mean proximity to nearest AQP4 hotspot (EU=Euclidean mm, SC=shortest-path units)")

plt.tight_layout()
fig.savefig(out_box, format="svg", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print("Saved:", out_box)

# lollipops
if (eu_obs_aqp1_all is None) or (eu_obs_aqp9_all is None):
    raise RuntimeError("EU npz missing obs_means_aqp1 / obs_means_aqp9.")
if (sc_obs_aqp1_all is None) or (sc_obs_aqp9_all is None):
    raise RuntimeError("SC npz missing obs_means_aqp1 / obs_means_aqp9.")

order_lolli = [p for p in order_left_to_right if (p in eu_idx) and (p in sc_idx)]
if len(order_lolli) == 0:
    raise RuntimeError("No overlapping phenotypes for lollipop plot.")

eu_obs_aqp4 = np.array([float(eu_obs[eu_idx[p]]) for p in order_lolli], float)
sc_obs_aqp4 = np.array([float(sc_obs[sc_idx[p]]) for p in order_lolli], float)

eu_obs_aqp1 = np.array([float(eu_obs_aqp1_all[eu_idx[p]]) for p in order_lolli], float)
eu_obs_aqp9 = np.array([float(eu_obs_aqp9_all[eu_idx[p]]) for p in order_lolli], float)

sc_obs_aqp1 = np.array([float(sc_obs_aqp1_all[sc_idx[p]]) for p in order_lolli], float)
sc_obs_aqp9 = np.array([float(sc_obs_aqp9_all[sc_idx[p]]) for p in order_lolli], float)

x_labels = [display_name.get(p, p) for p in order_lolli]

step = 0.75
x = np.arange(len(x_labels)) * step

off_metric = 0.18 * step
off_gene = 0.07 * step

gene_offsets = {"AQP4": -off_gene, "AQP1": 0.0, "AQP9": +off_gene}
x_eu = {g: x - off_metric + gene_offsets[g] for g in gene_offsets}
x_sc = {g: x + off_metric + gene_offsets[g] for g in gene_offsets}

gene_color = {"AQP4": blue, "AQP1": red, "AQP9": grey}

eu_vals = {"AQP4": eu_obs_aqp4, "AQP1": eu_obs_aqp1, "AQP9": eu_obs_aqp9}
sc_vals = {"AQP4": sc_obs_aqp4, "AQP1": sc_obs_aqp1, "AQP9": sc_obs_aqp9}

fig, ax = plt.subplots(figsize=(1.05 * len(x_labels) * step + 3.0, 5.2), dpi=300)

def draw_lollipops(ax, xs, vals, color, dashed=False):
    for xi, vi in zip(xs, vals):
        if not np.isfinite(vi):
            continue
        lc = ax.vlines(xi, 0, vi, colors=color, linewidth=1.5, zorder=2)
        if dashed:
            lc.set_linestyle((0, (2, 2)))
        ax.scatter([xi], [vi], s=34, color=color, linewidths=0.25, zorder=3)

# EU dashed, SC solid
for g in ["AQP4", "AQP1", "AQP9"]:
    draw_lollipops(ax, x_eu[g], eu_vals[g], gene_color[g], dashed=True)
for g in ["AQP4", "AQP1", "AQP9"]:
    draw_lollipops(ax, x_sc[g], sc_vals[g], gene_color[g], dashed=False)

ax.axhline(0, color="lightgray", linewidth=1.0)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=35, ha="right")
ax.set_xlim(x[0] - step * 0.7, x[-1] + step * 0.7)

vals_all = np.concatenate([
    eu_obs_aqp4, sc_obs_aqp4,
    eu_obs_aqp1, sc_obs_aqp1,
    eu_obs_aqp9, sc_obs_aqp9
])
vmin = float(np.nanmin(vals_all))
vmax = float(np.nanmax(vals_all))
pad = 0.12 * max(1e-6, (vmax - vmin))
ax.set_ylim(min(0, vmin - pad), vmax + pad)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("Observed mean proximity")

plt.tight_layout()
fig.savefig(out_lolli, format="svg", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print("Saved:", out_lolli)
