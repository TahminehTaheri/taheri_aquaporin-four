# SC shortest-path proximity 
# Saves .npz in /results and boxplot in /figures/proximity

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
from brainspace.null_models.moran import MoranRandomization
from bct import distance as bct_distance
from statsmodels.stats.multitest import multipletests

# helpers 
def pick_top_k(v, k):
    """Indices of the k largest finite values in v."""
    v = np.asarray(v, float)
    m = np.isfinite(v)
    if not m.any() or k <= 0:
        return np.array([], dtype=int)
    idx = np.where(m)[0]
    vv = v[m]
    k = min(k, vv.size)
    sel = np.argpartition(vv, -k)[-k:]
    return np.sort(idx[sel])

def pick_tail_k(v, k, tail="neg"):
    """Indices of the k smallest (neg tail) or k largest (pos tail) finite values."""
    v = np.asarray(v, float)
    m = np.isfinite(v)
    if not m.any() or k <= 0:
        return np.array([], dtype=int)
    idx = np.where(m)[0]
    vv = v[m]
    k = min(k, vv.size)
    if tail == "neg":
        sel = np.argpartition(vv, k - 1)[:k]
    else:  # "pos"
        sel = np.argpartition(vv, -k)[-k:]
    return np.sort(idx[sel])

def invdist_W(xyz):
    """Inverse-distance weight matrix for MoranRandomization."""
    D = distance_matrix(xyz, xyz).astype(float)
    np.fill_diagonal(D, np.inf)
    W = 1.0 / D
    W[~np.isfinite(W)] = 0.0
    return W

#  settings 
N = 400
TOP_PCT_GENE = 10      # hotspots (%): used for AQP4/AQP1/AQP9
TOP_PCT_EPI  = 10      # epicenters (%)
N_SURR = 10_000
SEED = 12

disease_order = ["EOA", "PS1", "3Rtau", "4Rtau", "TDP43A", "TDP43C", "DLB", "LOA"]

# paths 
root = Path("/Users/tahminehtaheri/aquaporin_four")

gene_dir = root / "data" / "AHBA_gene_expression" / "Schaefer400_MelbourneS4"
aqp4_fp = gene_dir / "aqp4_values.npy"
aqp1_fp = gene_dir / "aqp1_values.npy"
aqp9_fp = gene_dir / "aqp9_values.npy"

disease_csv = (
    root / "data" / "neurodegenerative_disease" / "Schaefer400_MelbourneS4"
    / f"atrophy_VBM_Tstat_parcellated_all_{N}.csv"
)
edema_fp = root / "data" / "edema" / "Schaefer400_MelbourneS4" / f"edema_1094_{N}.npy"

coords_fp = root / "data" / "coordination" / f"COG{N}_label.csv"
sc_fp = root / "data" / "SC" / "S1_sc_weighted400_individuals_omatrix-3_waytotal-standard_parc.npy"

res_dir = root / "results"
fig_dir = root / "figures" / "proximity"
res_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

out_npz = res_dir / "sc_proximity_moran_boxplot_data.npz"
out_fig = fig_dir / "sc_proximity_moran_boxplot_fdr8.svg"

# load data
aqp4 = np.load(aqp4_fp).astype(float).ravel()
aqp1 = np.load(aqp1_fp).astype(float).ravel()
aqp9 = np.load(aqp9_fp).astype(float).ravel()

disease_df = pd.read_csv(disease_csv)
Y_dis = disease_df[disease_order].to_numpy(float)  # keep sign

edema = np.load(edema_fp).astype(float).ravel()
xyz = pd.read_csv(coords_fp, header=None).iloc[:, :3].to_numpy(float)

SC = np.load(sc_fp).astype(float)

Ntot = aqp4.size
assert aqp1.size == Ntot and aqp9.size == Ntot
assert xyz.shape[0] == Ntot
assert Y_dis.shape[0] == Ntot
assert edema.size == Ntot
assert SC.shape[0] == Ntot and SC.shape[1] == Ntot


# SC shortest-path distances
L = np.where(SC > 0, 1.0 / (SC + 1e-12), np.inf)
Dsp, _ = bct_distance.distance_wei(L)


# define gene hotspots
k_hot4 = max(1, int(np.floor(TOP_PCT_GENE / 100.0 * np.isfinite(aqp4).sum())))
k_hot1 = max(1, int(np.floor(TOP_PCT_GENE / 100.0 * np.isfinite(aqp1).sum())))
k_hot9 = max(1, int(np.floor(TOP_PCT_GENE / 100.0 * np.isfinite(aqp9).sum())))

hot4 = pick_top_k(aqp4, k_hot4)
hot1 = pick_top_k(aqp1, k_hot1)
hot9 = pick_top_k(aqp9, k_hot9)


# Moran model
W = invdist_W(xyz)
mr = MoranRandomization(n_rep=N_SURR, random_state=SEED, tol=1e-8).fit(W)


# phenotypes
# Diseases use negative tail (most atrophy = most negative T)
# Edema uses positive tail
phenotypes = disease_order + ["Edema"]
tails = {nm: "neg" for nm in disease_order}
tails["Edema"] = "pos"


# run analysis
names = []
obs_aqp4 = []
obs_aqp1 = []
obs_aqp9 = []

p_raw = []
null_means_all = []

for i, nm in enumerate(phenotypes):
    vec = Y_dis[:, i] if nm != "Edema" else edema
    tail = tails[nm]


    nfin = int(np.isfinite(vec).sum())
    k_epi = max(1, int(np.floor(TOP_PCT_EPI / 100.0 * nfin)))
    epi = pick_tail_k(vec, k_epi, tail=tail)
    if epi.size == 0:
        print(f"[skip] {nm}: no epicenters")
        continue

    # observed mean distances to AQP4/AQP1/AQP9 hotspots
    def obs_mean_to_hot(epi_idx, hot_idx):
        d = np.min(Dsp[np.ix_(epi_idx, hot_idx)], axis=1)
        d = d[np.isfinite(d)]
        return np.nan if d.size == 0 else float(d.mean())

    m4 = obs_mean_to_hot(epi, hot4)
    m1 = obs_mean_to_hot(epi, hot1)
    m9 = obs_mean_to_hot(epi, hot9)

    if not np.isfinite(m4):
        print(f"[skip] {nm}: observed distances are all inf/NaN (disconnected SC?)")
        continue

    # Moran nulls ONLY for AQP4 analysis
    v = np.asarray(vec, float).copy()
    if not np.isfinite(v).all():
        v[~np.isfinite(v)] = np.nanmean(v[np.isfinite(v)])

    vec_surr = mr.randomize(v)  # (N_SURR, Ntot)

    null_mean = np.full(N_SURR, np.nan, float)
    for s in range(N_SURR):
        epi_s = pick_tail_k(vec_surr[s], k_epi, tail=tail)
        dn = np.min(Dsp[np.ix_(epi_s, hot4)], axis=1)
        dn = dn[np.isfinite(dn)]
        if dn.size:
            null_mean[s] = float(dn.mean())

    null_ok = null_mean[np.isfinite(null_mean)]
    p_att = (1 + np.sum(null_ok <= m4)) / (1 + null_ok.size)

    # store
    names.append(nm)
    obs_aqp4.append(m4)
    obs_aqp1.append(m1)
    obs_aqp9.append(m9)

    p_raw.append(float(p_att))
    null_means_all.append(null_mean)

    print(f"{nm:7s} | k_epi={k_epi:3d} | obs(AQP4)={m4:.4f} | p_raw={p_att:.3g}")


# pack arrays
names = np.array(names, dtype=object)
obs_aqp4 = np.array(obs_aqp4, float)
obs_aqp1 = np.array(obs_aqp1, float)
obs_aqp9 = np.array(obs_aqp9, float)

p_raw = np.array(p_raw, float)
null_means_all = np.vstack(null_means_all).astype(np.float32) if len(null_means_all) else np.zeros((0, N_SURR), np.float32)


# FDR on 8 diseases only (NOT Edema)
disease_set = set(disease_order)
is_disease = np.array([nm in disease_set for nm in names], dtype=bool)

q_fdr = np.full_like(p_raw, np.nan, dtype=float)
if is_disease.any():
    q_fdr[is_disease] = multipletests(p_raw[is_disease], method="fdr_bh")[1]


# results table (Spyder variable explorer)
df_sc = pd.DataFrame({
    "phenotype": names,
    "obs_mean_aqp4": obs_aqp4,
    "obs_mean_aqp1": obs_aqp1,
    "obs_mean_aqp9": obs_aqp9,
    "p_raw": p_raw,
    "q_fdr": q_fdr,  # NaN for Edema
})


# save NPZ 
np.savez(
    out_npz,
    phenotypes=names,
    obs_means=obs_aqp4,
    null_means=null_means_all,
    p_raw=p_raw,
    q_fdr=q_fdr,


    obs_means_aqp1=obs_aqp1,
    obs_means_aqp9=obs_aqp9,


    top_pct_gene=float(TOP_PCT_GENE),
    top_pct_epi=float(TOP_PCT_EPI),
    n_surr=int(N_SURR),
    seed=int(SEED),
)
print("Saved:", out_npz)


# plot 
null_for_plot = [null_means_all[i][np.isfinite(null_means_all[i])] for i in range(names.size)]
pos = np.arange(1, names.size + 1)

fig, ax = plt.subplots(figsize=(9, 1 + 0.9 * names.size), dpi=300)

bp = ax.boxplot(
    null_for_plot, vert=False, patch_artist=True,
    labels=[str(n) for n in names],
    showfliers=False, widths=0.6
)

for box in bp["boxes"]:
    box.set(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.25)
for k in ["whiskers", "caps", "medians"]:
    for line in bp[k]:
        line.set(color="tab:blue", linewidth=1.5)

ax.scatter(obs_aqp4, pos, s=55, color="tab:blue", edgecolor="white", linewidth=0.9, zorder=3)

xmax = max([np.nanmax(v) for v in null_for_plot] + [float(np.nanmax(obs_aqp4))])
ax.set_xlim(0, xmax * 1.22)

x_txt = ax.get_xlim()[1] * 0.985
for y0, nm, p0, q0 in zip(pos, names, p_raw, q_fdr):
    if nm in disease_set:
        star = "*" if (np.isfinite(q0) and q0 < 0.05) else ""
        ax.text(x_txt, y0, f"$q_\\mathrm{{FDR}}$={q0:.3g}{star}", va="center", ha="left", fontsize=10)
    else:
        star = "*" if p0 < 0.05 else ""
        ax.text(x_txt, y0, f"$p$={p0:.3g}{star}", va="center", ha="left", fontsize=10)

ax.set_xlabel("Mean shortest-path length to nearest AQP4 hotspot")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(out_fig, format="svg", dpi=300, bbox_inches="tight")
print("Saved:", out_fig)

plt.show()
plt.close(fig)
