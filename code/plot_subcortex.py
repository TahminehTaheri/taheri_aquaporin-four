##plot Tian S4 parcles

import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


def nan_vmin_vmax(x):
    x = np.asarray(x, float)
    return float(np.nanmin(x)), float(np.nanmax(x))


def plot_subcortex(data, atlas_rois, hemi_labels, cmap="OrRd",
                  vmin=None, vmax=None, outpath=None):

    views = {
        "Right Lateral": ("zy", (0, 0, -1)),
        "Left Lateral": ("yz", (0, 0, -1)),
        "Right Medial": ("yz", (0, 0, -1)),
        "Left Medial": ("zy", (0, 0, -1)),
    }

    plotters = []
    for view_name, (cam_pos, up_dir) in views.items():
        hemi = "R" if "Right" in view_name else "L"
        pl = pv.Plotter(off_screen=True, window_size=(512, 512))

        lo = np.nanmin(data) if vmin is None else vmin
        hi = np.nanmax(data) if vmax is None else vmax
        den = (hi - lo) if (hi - lo) != 0 else 1.0

        for i, (roi, hem) in enumerate(zip(atlas_rois, hemi_labels)):
            if hem != hemi:
                continue

            if np.isnan(data[i]):
                color = [0.7, 0.7, 0.7]
            else:
                color = cm.get_cmap(cmap)((data[i] - lo) / den)[:3]

            pl.add_mesh(roi, color=color)

        pl.camera_position = cam_pos
        pl.camera.up = up_dir
        plotters.append(pl)

        # plot 4 views
    imgs = [pl.screenshot(return_img=True) for pl in plotters]
    for pl in plotters:
        pl.close()

    # swap columns: (left <-> right) for both rows
    imgs = [imgs[1], imgs[0], imgs[3], imgs[2]]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.ravel()
    for ax, img in zip(axs, imgs):
        ax.imshow(img)
        ax.axis("off")


    # save
    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def build_tian_subcortex_meshes(atlas_fp, n_rois=54):
    atlas = pv.read(atlas_fp)
    rois = [
        atlas.image_threshold([i, i]).contour([1]).smooth_taubin(
            n_iter=25, pass_band=0.01, non_manifold_smoothing=True
        )
        for i in range(1, n_rois + 1)
    ]
    hemi = ["R" if i < 27 else "L" for i in range(len(rois))]
    return rois, hemi


# paths
N = 400
N_SUBCORTEX = 54
root = Path("/Users/tahminehtaheri/aquaporin_four")

atlas_fp = root / "data" / "atlases" / "Tian_Subcortex_S4_3T_1mm.nii.gz"
fig_dir = root / "figures" / "subcortex"

gene_dir = root / "data" / "AHBA_gene_expression" / "Schaefer400_MelbourneS4"
vascular_dir = root / "data" / "vascular_measures" / "Schaefer400_MelbourneS4"
edema_dir = root / "data" / "edema" / "Schaefer400_MelbourneS4"

# load atlas meshes
atlas_rois, hemi_labels = build_tian_subcortex_meshes(atlas_fp, n_rois=N_SUBCORTEX)

# load maps
aqp4 = np.load(gene_dir / "aqp4_values.npy").astype(float).ravel()
perfusion = np.load(vascular_dir / f"perfusion_{N}.npy").astype(float).ravel()
vdensity = np.load(vascular_dir / f"vdensity_{N}.npy").astype(float).ravel()
edema = np.load(edema_dir / f"edema_1094_{N}.npy").astype(float).ravel()

# keep subcortex only
aqp4_s = aqp4[:N_SUBCORTEX]
perfusion_s = perfusion[:N_SUBCORTEX]
vdensity_s = vdensity[:N_SUBCORTEX]
edema_s = edema[:N_SUBCORTEX]

# plot aqp4
vmin, vmax = nan_vmin_vmax(aqp4_s)
plot_subcortex(aqp4_s, atlas_rois, hemi_labels, cmap="OrRd",
              vmin=vmin, vmax=vmax, outpath=fig_dir / "surface_aqp4.png")

# plot edema
vmin, vmax = nan_vmin_vmax(edema_s)
plot_subcortex(edema_s, atlas_rois, hemi_labels, cmap="OrRd",
              vmin=vmin, vmax=vmax, outpath=fig_dir / "surface_edema.png")

# plot vein density
vabs = np.nanmax(np.abs(vdensity_s))
plot_subcortex(vdensity_s, atlas_rois, hemi_labels, cmap="coolwarm",
              vmin=-vabs, vmax=vabs, outpath=fig_dir / "surface_vdensity.png")

# plot perfusion
vabs = np.nanmax(np.abs(perfusion_s))
plot_subcortex(perfusion_s, atlas_rois, hemi_labels, cmap="coolwarm",
              vmin=-vabs, vmax=vabs, outpath=fig_dir / "surface_perfusion.png")
