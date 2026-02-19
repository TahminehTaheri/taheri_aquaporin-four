## whether inflammation refines the correspodence between aqp4 and neurodegeneration?

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
from brainspace.null_models.moran import MoranRandomization

# helpers
def invdist_W(xyz):
    D = distance_matrix(xyz, xyz).astype(float)
    np.fill_diagonal(D, np.inf)
    W = 1.0 / D
    W[~np.isfinite(W)] = 0.0
    return W

def fill_nan_with_mean(v):
    v = np.asarray(v, float).copy()
    if not np.isfinite(v).all():
        v[~np.isfinite(v)] = np.nanmean(v[np.isfinite(v)])
    return v

def load_or_make_moran_nulls(vec, xyz, out_fp, seed):
    if out_fp.exists():
        arr = np.load(out_fp).astype(float)
        return arr

    W = invdist_W(xyz)
    v = fill_nan_with_mean(vec)
    mr = MoranRandomization(n_rep=N_SURR, random_state=seed, tol=1e-8).fit(W)
    surr = mr.randomize(v).astype(np.float32)
    np.save(out_fp, surr)
    print("Saved Moran nulls:", out_fp)
    return surr

def diagonal_limits(x, y, pad=0.01):
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    d = (hi - lo) * pad
    return lo - d, hi + d

def scatter_r2(x, y, names, pvals, xlabel, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

    for i, nm in enumerate(names):
        color = "#de2d26" if pvals[i] < 0.05 else "grey"
        ax.scatter(x[i], y[i], s=95, color=color, edgecolors="black", linewidths=0.1, alpha=0.85)
        ax.text(x[i], y[i], nm, fontsize=9, ha="left", va="bottom")

    lo, hi = diagonal_limits(x, y)
    ax.plot([lo, hi], [lo, hi], color="gray", linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(outdir / fname, format="svg", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# settings
N = 400
N_SURR = 10_000
SEED = 12

disease_names = ["EOA", "PS1", "3Rtau", "4Rtau", "TDP43A", "TDP43C", "DLB", "LOA"]
marker_names = ["TSPO", "COX1", "COX2"]


# paths
root = Path("/Users/tahminehtaheri/aquaporin_four")

aqp4_fp = root / "data" / "AHBA_gene_expression" / "Schaefer400_MelbourneS4" / "aqp4_values.npy"

infl_dir = root / "data" / "inflammation_marker" / "Schaefer400_MelbourneS4"
tspo_fp = infl_dir / f"tspo_{N}.npy"
cox1_fp = infl_dir / f"cox1_{N}.npy"
cox2_fp = infl_dir / f"cox2_{N}.npy"

disease_csv = root / "data" / "neurodegenerative_disease" / "Schaefer400_MelbourneS4" / f"atrophy_VBM_Tstat_parcellated_all_{N}.csv"
edema_fp = root / "data" / "edema" / "Schaefer400_MelbourneS4" / f"edema_1094_{N}.npy"

coords_fp = root / "data" / "coordination" / f"COG{N}_label.csv"

res_dir = root / "results"
outdir = root / "figures" / "hierarchical_aqp4_base"
outdir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)

tspo_null_fp = res_dir / f"tspo_moran_nulls_{N_SURR}_seed{SEED}_N{N}.npy"
cox1_null_fp = res_dir / f"cox1_moran_nulls_{N_SURR}_seed{SEED}_N{N}.npy"
cox2_null_fp = res_dir / f"cox2_moran_nulls_{N_SURR}_seed{SEED}_N{N}.npy"

# load data
aqp4 = np.load(aqp4_fp).astype(float).ravel()

tspo = np.load(tspo_fp).astype(float).ravel()
cox1 = np.load(cox1_fp).astype(float).ravel()
cox2 = np.load(cox2_fp).astype(float).ravel()

df_dis = pd.read_csv(disease_csv)
disease = -df_dis[disease_names].to_numpy(float)

edema = np.load(edema_fp).astype(float).ravel()

xyz = pd.read_csv(coords_fp, header=None).iloc[:, :3].to_numpy(float)


# moran nulls: load from /results if available, else compute and save
tspo_surr = load_or_make_moran_nulls(tspo, xyz, tspo_null_fp, seed=SEED + 1)
cox1_surr = load_or_make_moran_nulls(cox1, xyz, cox1_null_fp, seed=SEED + 2)
cox2_surr = load_or_make_moran_nulls(cox2, xyz, cox2_null_fp, seed=SEED + 3)


# models
X_aqp4 = sm.add_constant(aqp4)
X_aqp4_tspo = sm.add_constant(np.column_stack([aqp4, tspo]))
X_aqp4_cox1 = sm.add_constant(np.column_stack([aqp4, cox1]))
X_aqp4_cox2 = sm.add_constant(np.column_stack([aqp4, cox2]))

n_dis = len(disease_names)

r2_aqp4 = np.zeros(n_dis)
r2_tspo = np.zeros(n_dis)
r2_cox1 = np.zeros(n_dis)
r2_cox2 = np.zeros(n_dis)

for i in range(n_dis):
    y = disease[:, i]
    r2_aqp4[i] = sm.OLS(y, X_aqp4).fit().rsquared_adj
    r2_tspo[i] = sm.OLS(y, X_aqp4_tspo).fit().rsquared_adj
    r2_cox1[i] = sm.OLS(y, X_aqp4_cox1).fit().rsquared_adj
    r2_cox2[i] = sm.OLS(y, X_aqp4_cox2).fit().rsquared_adj

delta_tspo = r2_tspo - r2_aqp4
delta_cox1 = r2_cox1 - r2_aqp4
delta_cox2 = r2_cox2 - r2_aqp4

delta_mat = np.vstack([delta_tspo, delta_cox1, delta_cox2])
delta_df = pd.DataFrame(delta_mat.T, columns=marker_names, index=disease_names)
print("\nΔR²_adj (AQP4+marker − AQP4) for diseases:")
print(delta_df.round(3))


# moran p-values for diseases
p_tspo = np.zeros(n_dis)
p_cox1 = np.zeros(n_dis)
p_cox2 = np.zeros(n_dis)

for i in range(n_dis):
    y = disease[:, i]
    r2_base = r2_aqp4[i]

    null_dt = np.empty(N_SURR)
    null_c1 = np.empty(N_SURR)
    null_c2 = np.empty(N_SURR)

    for r in range(N_SURR):
        X = sm.add_constant(np.column_stack([aqp4, tspo_surr[r]]))
        null_dt[r] = sm.OLS(y, X).fit().rsquared_adj - r2_base

        X = sm.add_constant(np.column_stack([aqp4, cox1_surr[r]]))
        null_c1[r] = sm.OLS(y, X).fit().rsquared_adj - r2_base

        X = sm.add_constant(np.column_stack([aqp4, cox2_surr[r]]))
        null_c2[r] = sm.OLS(y, X).fit().rsquared_adj - r2_base

    p_tspo[i] = (1 + np.sum(null_dt >= delta_tspo[i])) / (N_SURR + 1)
    p_cox1[i] = (1 + np.sum(null_c1 >= delta_cox1[i])) / (N_SURR + 1)
    p_cox2[i] = (1 + np.sum(null_c2 >= delta_cox2[i])) / (N_SURR + 1)

p_mat = np.vstack([p_tspo, p_cox1, p_cox2])
p_df = pd.DataFrame(p_mat.T, columns=marker_names, index=disease_names)
print("\nOne-sided Moran p-values for ΔR²_adj (diseases):")
print(p_df.round(4))


# edema: ΔR² + p-values
y_ed = edema
r2_base_ed = sm.OLS(y_ed, X_aqp4).fit().rsquared_adj
r2_tspo_ed = sm.OLS(y_ed, X_aqp4_tspo).fit().rsquared_adj
r2_cox1_ed = sm.OLS(y_ed, X_aqp4_cox1).fit().rsquared_adj
r2_cox2_ed = sm.OLS(y_ed, X_aqp4_cox2).fit().rsquared_adj

delta_tspo_ed = r2_tspo_ed - r2_base_ed
delta_cox1_ed = r2_cox1_ed - r2_base_ed
delta_cox2_ed = r2_cox2_ed - r2_base_ed

null_dt = np.empty(N_SURR)
null_c1 = np.empty(N_SURR)
null_c2 = np.empty(N_SURR)

for r in range(N_SURR):
    X = sm.add_constant(np.column_stack([aqp4, tspo_surr[r]]))
    null_dt[r] = sm.OLS(y_ed, X).fit().rsquared_adj - r2_base_ed

    X = sm.add_constant(np.column_stack([aqp4, cox1_surr[r]]))
    null_c1[r] = sm.OLS(y_ed, X).fit().rsquared_adj - r2_base_ed

    X = sm.add_constant(np.column_stack([aqp4, cox2_surr[r]]))
    null_c2[r] = sm.OLS(y_ed, X).fit().rsquared_adj - r2_base_ed

p_tspo_ed = (1 + np.sum(null_dt >= delta_tspo_ed)) / (N_SURR + 1)
p_cox1_ed = (1 + np.sum(null_c1 >= delta_cox1_ed)) / (N_SURR + 1)
p_cox2_ed = (1 + np.sum(null_c2 >= delta_cox2_ed)) / (N_SURR + 1)

edema_results = pd.DataFrame({
    "marker": marker_names,
    "delta_R2_adj": [delta_tspo_ed, delta_cox1_ed, delta_cox2_ed],
    "p_MSR": [p_tspo_ed, p_cox1_ed, p_cox2_ed],
})
print("\nEdema ΔR²_adj (AQP4+marker − AQP4) and Moran p-values:")
print(edema_results.round(4))


# edema boxplot
null_data_ed = [null_dt, null_c1, null_c2]
obs_delta_ed = [delta_tspo_ed, delta_cox1_ed, delta_cox2_ed]
p_ed = [p_tspo_ed, p_cox1_ed, p_cox2_ed]

blue = "#1f77b4"

fig, ax = plt.subplots(figsize=(7, 3.2), dpi=300)
bp = ax.boxplot(
    null_data_ed,
    vert=False,
    patch_artist=True,
    labels=marker_names,
    showfliers=False,
    widths=0.6,
    medianprops=dict(color=blue, linewidth=2),
)
for box in bp["boxes"]:
    box.set(facecolor=blue, edgecolor=blue, alpha=0.35)
for k in ["whiskers", "caps"]:
    for line in bp[k]:
        line.set(color=blue, linewidth=1.5)

pos = np.arange(1, len(marker_names) + 1)
ax.scatter(obs_delta_ed, pos, s=60, color=blue, edgecolor="white", linewidth=0.9, zorder=3)

min_x = min([np.nanmin(d) for d in null_data_ed] + obs_delta_ed)
max_x = max([np.nanmax(d) for d in null_data_ed] + obs_delta_ed)
pad = 0.15 * (max_x - min_x)
ax.set_xlim(min_x - pad, max_x + pad)

for y, p in zip(pos, p_ed):
    star = " *" if p < 0.05 else ""
    ax.text(1.01, y, f"p={p:.4f}{star}", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel(r"$\Delta R^2_{\mathrm{adj}}$ (AQP4+marker − AQP4)")
ax.set_title("Edema: null ΔR² and observed ΔR²")

plt.tight_layout()
fig.savefig(outdir / "boxplot_delta_r2_edema_markers.svg", format="svg", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)


# heatmap (diseases only)
fig, ax = plt.subplots(figsize=(8, 3), dpi=300)

vmax = float(np.nanmax(delta_mat))
im = ax.imshow(delta_mat, aspect="auto", cmap="coolwarm", vmin=0, vmax=vmax)

ax.set_xticks(np.arange(n_dis))
ax.set_xticklabels(disease_names, rotation=45, ha="right")
ax.set_yticks(np.arange(len(marker_names)))
ax.set_yticklabels(marker_names)

for i in range(len(marker_names)):
    for j in range(n_dis):
        txt = f"{delta_mat[i, j]:.2f}"
        if p_mat[i, j] < 0.05:
            txt += "*"
        ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=9)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r"$\Delta R^2_{\mathrm{adj}}$ (AQP4+marker − AQP4)")

ax.set_title(r"Gain in $R^2_{\mathrm{adj}}$ from adding inflammatory markers to AQP4")
plt.tight_layout()
fig.savefig(outdir / "heatmap_delta_r2_all_markers_AQP4base.svg", format="svg", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)


# scatter plots (diseases only)
scatter_r2(
    r2_aqp4, r2_tspo, disease_names, p_tspo,
    r"$R^2_{\mathrm{adj}}$ (AQP4 only)",
    r"$R^2_{\mathrm{adj}}$ (AQP4 + TSPO)",
    "Disease prediction: AQP4 vs AQP4+TSPO",
    "scatter_r2_AQP4_vs_AQP4_TSPO.svg",
)

scatter_r2(
    r2_aqp4, r2_cox1, disease_names, p_cox1,
    r"$R^2_{\mathrm{adj}}$ (AQP4 only)",
    r"$R^2_{\mathrm{adj}}$ (AQP4 + COX1)",
    "Disease prediction: AQP4 vs AQP4+COX1",
    "scatter_r2_AQP4_vs_AQP4_COX1.svg",
)

scatter_r2(
    r2_aqp4, r2_cox2, disease_names, p_cox2,
    r"$R^2_{\mathrm{adj}}$ (AQP4 only)",
    r"$R^2_{\mathrm{adj}}$ (AQP4 + COX2)",
    "Disease prediction: AQP4 vs AQP4+COX2",
    "scatter_r2_AQP4_vs_AQP4_COX2.svg",
)
