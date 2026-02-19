from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree, distance_matrix
from brainspace.null_models.moran import MoranRandomization
from statsmodels.stats.multitest import multipletests

# helpers
def pick_top_k(v, k):
    v = np.asarray(v, float)
    m = np.isfinite(v)
    if not m.any() or k <= 0:
        return np.array([], dtype=int)
    idx = np.where(m)[0]
    vv = v[m]
    k = min(k, vv.size)
    sel = np.argpartition(vv, -k)[-k:]
    return np.sort(idx[sel])

# settings
N = 400
TOP_PCT_GENE = 10
TOP_PCT_EPI = 10
N_SURR = 10_000
SEED = 12

disease_order = ["EOA", "PS1", "3Rtau", "4Rtau", "TDP43A", "TDP43C", "DLB", "LOA"]


# paths
root = Path("/Users/tahminehtaheri/aquaporin_four")

gene_dir = root / "data" / "AHBA_gene_expression" / "Schaefer400_MelbourneS4"
aqp4_fp = gene_dir / "aqp4_values.npy"
aqp1_fp = gene_dir / "aqp1_values.npy"
aqp9_fp = gene_dir / "aqp9_values.npy"

coords_fp = root / "data" / "coordination" / f"COG{N}_label_2mm.csv"

disease_csv = (
    root / "data" / "neurodegenerative_disease" / "Schaefer400_MelbourneS4"
    / f"atrophy_VBM_Tstat_parcellated_all_{N}.csv"
)
edema_fp = root / "data" / "edema" / "Schaefer400_MelbourneS4" / f"edema_1094_{N}.npy"

res_dir = root / "results"
fig_dir = root / "figures"

out_npz = res_dir / "euclid_proximity_moran_boxplot_data.npz"
out_fig = fig_dir / "euclid_proximity_moran_boxplot_fdr8.svg"

# load
aqp4 = np.load(aqp4_fp).astype(float).ravel()
aqp1 = np.load(aqp1_fp).astype(float).ravel()
aqp9 = np.load(aqp9_fp).astype(float).ravel()

xyz = pd.read_csv(coords_fp, header=None).iloc[:, :3].to_numpy(float)

dis_df = pd.read_csv(disease_csv)
diseases = dis_df[disease_order].to_numpy(float)
edema = np.load(edema_fp).astype(float).ravel()

Ntot = aqp4.size
assert aqp1.size == Ntot and aqp9.size == Ntot
assert xyz.shape[0] == Ntot
assert diseases.shape[0] == Ntot
assert edema.size == Ntot


# hotspots (AQP4/AQP1/AQP9)
k_hot4 = max(1, int(np.floor(TOP_PCT_GENE / 100.0 * np.isfinite(aqp4).sum())))
k_hot1 = max(1, int(np.floor(TOP_PCT_GENE / 100.0 * np.isfinite(aqp1).sum())))
k_hot9 = max(1, int(np.floor(TOP_PCT_GENE / 100.0 * np.isfinite(aqp9).sum())))

hot4 = pick_top_k(aqp4, k_hot4)
hot1 = pick_top_k(aqp1, k_hot1)
hot9 = pick_top_k(aqp9, k_hot9)

tree4 = cKDTree(xyz[hot4])
tree1 = cKDTree(xyz[hot1])
tree9 = cKDTree(xyz[hot9])


# Moran weights
D = distance_matrix(xyz, xyz).astype(float)
np.fill_diagonal(D, np.inf)
W = 1.0 / D
W[~np.isfinite(W)] = 0.0


# run (8 diseases + edema)
names = disease_order + ["Edema"]

p_raw = []
q_fdr = []
obs_aqp4 = []
obs_aqp1 = []
obs_aqp9 = []
null_means = []

for i, nm in enumerate(names):
    v = diseases[:, i] if nm != "Edema" else edema
    tail = "neg" if nm != "Edema" else "pos"

    m = np.isfinite(v)
    k_epi = max(1, int(np.floor(TOP_PCT_EPI / 100.0 * m.sum())))
    idx = np.where(m)[0]
    vv = v[m]

    if tail == "neg":
        epi_sel = np.argpartition(vv, k_epi - 1)[:k_epi]
    else:
        epi_sel = np.argpartition(vv, -k_epi)[-k_epi:]
    epi_idx = np.sort(idx[epi_sel])

    d4, _ = tree4.query(xyz[epi_idx], k=1)
    d1, _ = tree1.query(xyz[epi_idx], k=1)
    d9, _ = tree9.query(xyz[epi_idx], k=1)

    m4 = float(np.mean(d4))
    m1 = float(np.mean(d1))
    m9 = float(np.mean(d9))

    obs_aqp4.append(m4)
    obs_aqp1.append(m1)
    obs_aqp9.append(m9)

    v_fill = v.astype(float).copy()
    if not np.isfinite(v_fill).all():
        v_fill[~np.isfinite(v_fill)] = np.nanmean(v_fill[np.isfinite(v_fill)])

    mr = MoranRandomization(n_rep=N_SURR, random_state=SEED + i, tol=1e-8).fit(W)
    v_surr = mr.randomize(v_fill)

    nm_vec = np.empty(N_SURR, float)
    for r in range(N_SURR):
        vs = v_surr[r]
        m2 = np.isfinite(vs)
        idx2 = np.where(m2)[0]
        vv2 = vs[m2]

        k = min(k_epi, int(m2.sum()))
        if tail == "neg":
            sel2 = np.argpartition(vv2, k - 1)[:k]
        else:
            sel2 = np.argpartition(vv2, -k)[-k:]
        epi2 = np.sort(idx2[sel2])

        dn, _ = tree4.query(xyz[epi2], k=1)
        nm_vec[r] = float(np.mean(dn))

    null_means.append(nm_vec)

    p_att = (1 + np.sum(nm_vec <= m4)) / (1 + nm_vec.size)
    p_raw.append(float(p_att))

p_raw = np.array(p_raw, float)
obs_aqp4 = np.array(obs_aqp4, float)
obs_aqp1 = np.array(obs_aqp1, float)
obs_aqp9 = np.array(obs_aqp9, float)
null_means = np.vstack(null_means).astype(np.float32)


# FDR only on 8 diseases
is_disease = np.array([nm in set(disease_order) for nm in names], dtype=bool)
q_fdr = np.full_like(p_raw, np.nan, dtype=float)
_, qvals_dis, _, _ = multipletests(p_raw[is_disease], method="fdr_bh")
q_fdr[is_disease] = qvals_dis


# table 
df = pd.DataFrame({
    "phenotype": names,
    "obs_mean_aqp4": obs_aqp4,
    "obs_mean_aqp1": obs_aqp1,
    "obs_mean_aqp9": obs_aqp9,
    "p_raw": p_raw,
    "q_fdr": q_fdr,
})


# save
res_dir.mkdir(parents=True, exist_ok=True)
np.savez(
    out_npz,
    phenotypes=np.array(names, dtype=object),
    obs_means=obs_aqp4,
    null_means=null_means,
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


# plot (AQP4 only)
fig_dir.mkdir(parents=True, exist_ok=True)

labels_y = [f"{nm}" for nm in names]
pos = np.arange(1, len(names) + 1)

fig, ax = plt.subplots(figsize=(9, 1 + 0.85 * len(names)), dpi=300)
bp = ax.boxplot(null_means.T, vert=False, patch_artist=True, labels=labels_y, showfliers=False)

blue = "#1f77b4"
for box in bp["boxes"]:
    box.set(facecolor=blue, edgecolor=blue, alpha=0.35)
for med in bp["medians"]:
    med.set(color=blue, linewidth=2)
for w in bp["whiskers"]:
    w.set(color=blue, linewidth=1.5)
for cap in bp["caps"]:
    cap.set(color=blue, linewidth=1.5)

ax.scatter(obs_aqp4, pos, s=55, color=blue, edgecolor="white", linewidth=0.9, zorder=3)

max_x = float(np.nanmax(null_means))
ax.set_xlim(0, max(max_x, float(np.max(obs_aqp4))) * 1.18)

x_txt = ax.get_xlim()[1] * 0.985
for y0, nm, p0, q0 in zip(pos, names, p_raw, q_fdr):
    if nm in set(disease_order):
        star = "*" if q0 < 0.05 else ""
        ax.text(x_txt, y0, f"$q_\\mathrm{{FDR}}$={q0:.3g}{star}", va="center", ha="left", fontsize=10)
    else:
        star = "*" if p0 < 0.05 else ""
        ax.text(x_txt, y0, f"$p$={p0:.3g}{star}", va="center", ha="left", fontsize=10)

ax.set_xlabel("Mean Euclidean distance to nearest AQP4 hotspot (mm)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(out_fig, format="svg", dpi=300, bbox_inches="tight")
print("Saved:", out_fig)

plt.show()
plt.close(fig)
