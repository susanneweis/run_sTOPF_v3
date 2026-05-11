import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nilearn import datasets, plotting, surface

# Use the existing utility plotting style for glass brains and slices.
# Keep _util_glass_brains.py in the same folder as this script,
# or somewhere on your PYTHONPATH.
from _util_glass_brains import create_glassbrains


# ---------------------------------------------------------------------
# Network definitions
# ---------------------------------------------------------------------
NETWORKS_17 = [
    "VisCent",
    "VisPeri",
    "SomMotA",
    "SomMotB",
    "DorsAttnA",
    "DorsAttnB",
    "SalVentAttnA",
    "SalVentAttnB",
    "LimbicB",
    "LimbicA",
    "ContA",
    "ContB",
    "ContC",
    "DefaultA",
    "DefaultB",
    "DefaultC",
    "TempPar",
]


def assign_network(region_name):
    """
    Assign each region to one of the 17 cortical networks.
    If none of the network names occurs in the region name,
    assign it to Subcortical.
    """
    region_name = str(region_name)

    for net in NETWORKS_17:
        if net in region_name:
            return net

    return "Subcortical"


# ---------------------------------------------------------------------
# ROI lookup helper
# ---------------------------------------------------------------------
def _roi_names_to_lookup(roi_names):
    """
    Convert roi_names into a flexible lookup table.

    Supported inputs:
    - list/tuple/array of ROI names:
        atlas label is assumed to be position + 1
    - pandas DataFrame with one ROI-name column and optionally one index/id column
    - path to csv/tsv file containing ROI names and optionally atlas labels

    The returned dict maps string ROI names -> atlas integer labels.
    """

    if isinstance(roi_names, str):
        if roi_names.endswith(".tsv"):
            roi_names = pd.read_csv(roi_names, sep="\t")
        else:
            roi_names = pd.read_csv(roi_names)

    if isinstance(roi_names, pd.DataFrame):
        cols = list(roi_names.columns)

        name_candidates = [
            c for c in cols
            if c.lower() in ["region", "roi", "roi_name", "name", "label", "labels"]
        ]
        name_col = name_candidates[0] if name_candidates else cols[0]

        id_candidates = [
            c for c in cols
            if c.lower() in ["index", "idx", "id", "label_id", "atlas_id", "roi_id", "value"]
        ]

        if id_candidates:
            id_col = id_candidates[0]
            return {
                str(row[name_col]): int(row[id_col])
                for _, row in roi_names[[name_col, id_col]].dropna().iterrows()
            }
        else:
            return {
                str(name): i + 1
                for i, name in enumerate(roi_names[name_col].dropna().tolist())
            }

    return {str(name): i + 1 for i, name in enumerate(list(roi_names))}


# ---------------------------------------------------------------------
# Build value image helper
# ---------------------------------------------------------------------
def make_value_img_from_region_table(
    value_file,
    value_name,
    value_roi_name,
    roi_names,
    at_path,
    name="",
):
    """
    Convert a region-wise value table into a NIfTI image.
    """

    df = pd.read_csv(value_file)
    atlas_img = nib.load(at_path)
    atlas_data = atlas_img.get_fdata()

    value_img_data = np.zeros(atlas_data.shape, dtype=float)

    region_numeric = pd.to_numeric(df[value_roi_name], errors="coerce")

    if region_numeric.notna().all():
        for _, row in df.iterrows():
            roi_id = int(row[value_roi_name])
            val = row[value_name]
            if pd.notna(val):
                value_img_data[atlas_data == roi_id] = float(val)
    else:
        roi_lookup = _roi_names_to_lookup(roi_names)
        missing = []

        for _, row in df.iterrows():
            region = str(row[value_roi_name])
            val = row[value_name]

            if pd.isna(val):
                continue

            if region not in roi_lookup:
                missing.append(region)
                continue

            roi_id = int(roi_lookup[region])
            value_img_data[atlas_data == roi_id] = float(val)

        if len(missing) > 0:
            missing_unique = sorted(set(missing))
            print(
                f"WARNING for {name}: {len(missing_unique)} regions were not found "
                f"in roi_names and were left as zero. First missing regions: "
                f"{missing_unique[:10]}"
            )

    value_img = nib.Nifti1Image(value_img_data, atlas_img.affine, atlas_img.header)

    return value_img, value_img_data


# ---------------------------------------------------------------------
# Brain plotting helpers
# ---------------------------------------------------------------------
def _get_symmetric_limits(value_img_data):
    """
    Get symmetric plotting limits around zero from non-zero finite values.
    """

    nonzero_vals = value_img_data[np.isfinite(value_img_data) & (value_img_data != 0)]

    if len(nonzero_vals) == 0:
        vmax = 1.0
        vmin = -1.0
    else:
        max_abs = np.nanmax(np.abs(nonzero_vals))
        vmax = float(max_abs)
        vmin = float(-max_abs)

    return vmin, vmax


def create_3d_brain_regions_from_img(
    value_img,
    value_img_data,
    title_str,
    out_path,
    name,
    cmap_mode="continuous",
):
    """
    Create 3D surface-brain visualisations from an already-created NIfTI image.

    Outputs:
    - out_path/3d_brains/<name>_3D_surface.png
    - out_path/3d_brains_html/<name>_3D_interactive.html
    """

    png_dir = os.path.join(out_path, "3d_brains")
    html_dir = os.path.join(out_path, "3d_brains_html")

    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(html_dir, exist_ok=True)

    png_file = os.path.join(png_dir, f"{name}_3D_surface.png")
    html_file = os.path.join(html_dir, f"{name}_3D_interactive.html")

    vmin, vmax = _get_symmetric_limits(value_img_data)

    cmap = "cold_hot" if cmap_mode == "continuous" else "tab20"

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    texture_left = surface.vol_to_surf(value_img, fsaverage.pial_left)
    texture_right = surface.vol_to_surf(value_img, fsaverage.pial_right)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title_str, fontsize=14)

    views = [
        ("left", "lateral", fsaverage.infl_left, texture_left, fsaverage.sulc_left),
        ("left", "medial", fsaverage.infl_left, texture_left, fsaverage.sulc_left),
        ("right", "lateral", fsaverage.infl_right, texture_right, fsaverage.sulc_right),
        ("right", "medial", fsaverage.infl_right, texture_right, fsaverage.sulc_right),
    ]

    for i, (hemi, view, surf_mesh, texture, bg_map) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")

        plotting.plot_surf_stat_map(
            surf_mesh=surf_mesh,
            stat_map=texture,
            hemi=hemi,
            view=view,
            bg_map=bg_map,
            axes=ax,
            colorbar=(i == 4),
            title=f"{hemi} {view}",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            threshold=None,
        )

    plt.tight_layout()
    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    try:
        view = plotting.view_img_on_surf(
            value_img,
            surf_mesh="fsaverage5",
            cmap=cmap,
            symmetric_cmap=True,
            vmax=vmax,
            threshold=None,
            title=title_str,
        )
        view.save_as_html(html_file)
    except Exception as exc:
        print(f"WARNING: Could not create interactive HTML for {name}: {exc}")


def create_glassbrain_regions_from_img(
    value_img,
    value_img_data,
    title_str,
    out_path,
    name,
):
    """
    Create glass-brain visualisations from an already-created NIfTI image.

    Outputs:
    - out_path/glassbrains/glassbrain_<name>.png
    - out_path/slices/slices_<name>.png
    """

    png_dir = os.path.join(out_path, "glass_brains")
    os.makedirs(png_dir, exist_ok=True)

    png_file = os.path.join(png_dir, f"{name}_glassbrain.png")

    _, vmax = _get_symmetric_limits(value_img_data)

    display = plotting.plot_glass_brain(
        value_img,
        display_mode="lyrz",
        colorbar=True,
        cmap="cold_hot",
        symmetric_cbar=True,
        vmax=vmax,
        threshold=None,
        title=title_str,
        plot_abs=False,
    )

    display.savefig(png_file, dpi=300)
    display.close()


def create_all_brains(
    value_file,
    value_name,
    value_roi_name,
    roi_names,
    at_path,
    title_str,
    out_path,
    name,
):
    """
    Create the NIfTI value image once, save it once, and reuse it for both
    3D surface and glass-brain visualisations.

    Outputs:
    - out_path/value_images/<name>_values.nii.gz
    - out_path/3d_brains/<name>_3D_surface.png
    - out_path/3d_brains_html/<name>_3D_interactive.html
    - out_path/glassbrains/glassbrain_<name>.png
    - out_path/slices/slices_<name>.png
    """

    value_img, value_img_data = make_value_img_from_region_table(
        value_file=value_file,
        value_name=value_name,
        value_roi_name=value_roi_name,
        roi_names=roi_names,
        at_path=at_path,
        name=name,
    )

    img_dir = os.path.join(out_path, "value_images")
    os.makedirs(img_dir, exist_ok=True)

    nii_file = os.path.join(img_dir, f"{name}_values.nii.gz")
    nib.save(value_img, nii_file)

    create_3d_brain_regions_from_img(
        value_img=value_img,
        value_img_data=value_img_data,
        title_str=title_str,
        out_path=out_path,
        name=name,
        cmap_mode="continuous",
    )

    # Use the utility glass-brain function instead of the local nilearn style.
    # This also creates matching slice plots.
    create_glassbrains(
        value_file=value_file,
        value_name=value_name,
        value_roi_name=value_roi_name,
        roi_names=roi_names,
        at_path=at_path,
        title_str=title_str,
        out_path=out_path,
        name=name,
        cmap_mode="continuous",
    )


# ---------------------------------------------------------------------
# Helper: convert network scores back to region scores for plotting
# ---------------------------------------------------------------------
def expand_network_scores_to_regions(df_region_network, df_network_scores, score_col):
    """
    Takes one score per network and assigns that score back to all regions
    belonging to that network.

    This is necessary because the atlas is still region-wise.
    The plotted brain will therefore show all regions in one network with
    the same network-average value.
    """

    expanded = df_region_network.merge(
        df_network_scores[["Network", score_col]],
        on="Network",
        how="left",
    )

    return expanded[["Region", "Network", score_col]].sort_values("Region")


# ---------------------------------------------------------------------
# Main analysis: network-wise version with 3D brains and glass brains
# ---------------------------------------------------------------------
def main(base_path, proj, code, roi_names, at_path):

    results_path = f"{base_path}/results_run_sTOPF_{code}_data_{proj}"
    res_in_dir = f"{results_path}/sex_separability"

    out_dir = f"{res_in_dir}/movies_networks"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(f"{res_in_dir}/ALL_movies_sex_diff_scores.csv")
    df = df[~df["movie"].isin(["REST1", "REST2", "concat"])].copy()

    # --------------------------------------------------------------
    # Assign every region to one of the 17 networks or Subcortical
    # --------------------------------------------------------------
    df["Network"] = df["Region"].apply(assign_network)

    region_network_lookup = (
        df[["Region", "Network"]]
        .drop_duplicates()
        .sort_values(["Network", "Region"])
    )

    region_network_lookup.to_csv(
        os.path.join(out_dir, "region_to_network_lookup.csv"),
        index=False,
    )

    # --------------------------------------------------------------
    # Network-wise movie sensitivity
    # --------------------------------------------------------------
    network_movie_scores = (
        df.groupby(["movie", "Network"], as_index=False)["score"]
        .mean()
    )

    network_movie_scores.to_csv(
        os.path.join(out_dir, "network_by_movie_scores_long.csv"),
        index=False,
    )

    network_sens = (
        network_movie_scores.groupby("Network")["score"]
        .agg(
            mean_score="mean",
            sd_across_movies="std",
            min_score="min",
            max_score="max",
            range_across_movies=lambda x: x.max() - x.min(),
            n_movies="count",
        )
        .reset_index()
        .sort_values("sd_across_movies", ascending=False)
    )

    network_sens.to_csv(
        os.path.join(out_dir, "network_movie_sensitivity_descriptive.csv"),
        index=False,
    )

    # --------------------------------------------------------------
    # Mean sex separability across movies, summarized by network
    # --------------------------------------------------------------
    mean_network = (
        network_movie_scores.groupby("Network", as_index=False)["score"]
        .mean()
        .rename(columns={"score": "mean_score_across_movies"})
        .sort_values("mean_score_across_movies", ascending=False)
    )

    mean_network_file = os.path.join(
        out_dir,
        "mean_sex_separability_across_movies_by_network.csv",
    )
    mean_network.to_csv(mean_network_file, index=False)

    mean_network_expanded = expand_network_scores_to_regions(
        df_region_network=region_network_lookup,
        df_network_scores=mean_network,
        score_col="mean_score_across_movies",
    )

    mean_network_expanded_file = os.path.join(
        out_dir,
        "mean_sex_separability_across_movies_by_network_EXPANDED_TO_REGIONS.csv",
    )
    mean_network_expanded.to_csv(mean_network_expanded_file, index=False)

    create_all_brains(
        value_file=mean_network_expanded_file,
        value_name="mean_score_across_movies",
        value_roi_name="Region",
        roi_names=roi_names,
        at_path=at_path,
        title_str="Mean sex separability across movies by network",
        out_path=out_dir,
        name="mean_sex_separability_score_across_movies_by_network",
    )

    # --------------------------------------------------------------
    # Movie-wise overall sex-difference strength, network-wise
    # --------------------------------------------------------------
    movie_summary = (
        network_movie_scores.groupby("movie")["score"]
        .agg(
            mean_score="mean",
            median_score="median",
            sd_across_networks="std",
            max_score="max",
            n_networks="count",
        )
        .reset_index()
        .sort_values("mean_score", ascending=False)
    )

    movie_summary.to_csv(
        os.path.join(out_dir, "movie_summary_scores_by_network.csv"),
        index=False,
    )

    # --------------------------------------------------------------
    # Network x movie matrix
    # --------------------------------------------------------------
    matrix = network_movie_scores.pivot(
        index="Network",
        columns="movie",
        values="score",
    )

    matrix.to_csv(
        os.path.join(out_dir, "network_by_movie_score_matrix.csv")
    )

    # --------------------------------------------------------------
    # 3D brains and glass brains: one network-wise map per movie
    # --------------------------------------------------------------
    per_movie_dir = os.path.join(out_dir, "per_movie_network_score_tables")
    os.makedirs(per_movie_dir, exist_ok=True)

    movies = sorted(df["movie"].dropna().unique())
    movies = [m for m in movies if m not in ["REST1", "REST2", "concat"]]

    for movie in movies:

        movie_network_df = (
            network_movie_scores[network_movie_scores["movie"] == movie]
            .copy()
            .sort_values("Network")
        )

        movie_safe = str(movie).replace("/", "_").replace(" ", "_")

        movie_network_file = os.path.join(
            per_movie_dir,
            f"sex_separability_score_by_network_{movie_safe}.csv",
        )

        movie_network_df.to_csv(movie_network_file, index=False)

        movie_network_expanded = expand_network_scores_to_regions(
            df_region_network=region_network_lookup,
            df_network_scores=movie_network_df,
            score_col="score",
        )

        movie_expanded_file = os.path.join(
            per_movie_dir,
            f"sex_separability_score_by_network_{movie_safe}_EXPANDED_TO_REGIONS.csv",
        )

        movie_network_expanded.to_csv(movie_expanded_file, index=False)

        create_all_brains(
            value_file=movie_expanded_file,
            value_name="score",
            value_roi_name="Region",
            roi_names=roi_names,
            at_path=at_path,
            title_str=f"Sex separability by network - {movie}",
            out_path=out_dir,
            name=f"sex_separability_score_by_network_{movie_safe}",
        )

    # --------------------------------------------------------------
    # MOVIE-SPECIFIC NETWORKS
    # Networks unusually high/low relative to their own across-movie mean
    # --------------------------------------------------------------
    specific_dir = os.path.join(out_dir, "movie_specific_networks_1SD")
    os.makedirs(specific_dir, exist_ok=True)

    network_stats = (
        network_movie_scores.groupby("Network")["score"]
        .agg(network_mean="mean", network_sd="std")
        .reset_index()
    )

    df_specific = network_movie_scores.merge(
        network_stats,
        on="Network",
        how="left",
    )

    df_specific["network_sd"] = df_specific["network_sd"].replace(0, pd.NA)

    df_specific["z_score"] = (
        (df_specific["score"] - df_specific["network_mean"])
        / df_specific["network_sd"]
    )

    df_specific.to_csv(
        os.path.join(specific_dir, "network_movie_z_scores.csv"),
        index=False,
    )

    for movie in movies:

        movie_df = df_specific[df_specific["movie"] == movie].copy()
        movie_safe = str(movie).replace("/", "_").replace(" ", "_")

        # ----------------------------------------------------------
        # BIG NETWORK EFFECTS > +1 SD
        # ----------------------------------------------------------
        big_df = movie_df.copy()
        big_df.loc[big_df["z_score"] <= 1, "score"] = 0

        big_network_file = os.path.join(
            specific_dir,
            f"movie_specific_big_network_effects_1SD_{movie_safe}.csv",
        )

        big_df.to_csv(big_network_file, index=False)

        big_expanded = expand_network_scores_to_regions(
            df_region_network=region_network_lookup,
            df_network_scores=big_df,
            score_col="score",
        )

        big_expanded_file = os.path.join(
            specific_dir,
            f"movie_specific_big_network_effects_1SD_{movie_safe}_EXPANDED_TO_REGIONS.csv",
        )

        big_expanded.to_csv(big_expanded_file, index=False)

        create_all_brains(
            value_file=big_expanded_file,
            value_name="score",
            value_roi_name="Region",
            roi_names=roi_names,
            at_path=at_path,
            title_str=f"Movie-specific HIGH network effects (>1 SD) - {movie}",
            out_path=specific_dir,
            name=f"movie_specific_big_network_effects_1SD_{movie_safe}",
        )

        # ----------------------------------------------------------
        # SMALL NETWORK EFFECTS < -1 SD
        # ----------------------------------------------------------
        small_df = movie_df.copy()
        small_df.loc[small_df["z_score"] >= -1, "score"] = 0

        small_network_file = os.path.join(
            specific_dir,
            f"movie_specific_small_network_effects_1SD_{movie_safe}.csv",
        )

        small_df.to_csv(small_network_file, index=False)

        small_expanded = expand_network_scores_to_regions(
            df_region_network=region_network_lookup,
            df_network_scores=small_df,
            score_col="score",
        )

        small_expanded_file = os.path.join(
            specific_dir,
            f"movie_specific_small_network_effects_1SD_{movie_safe}_EXPANDED_TO_REGIONS.csv",
        )

        small_expanded.to_csv(small_expanded_file, index=False)

        create_all_brains(
            value_file=small_expanded_file,
            value_name="score",
            value_roi_name="Region",
            roi_names=roi_names,
            at_path=at_path,
            title_str=f"Movie-specific LOW network effects (<-1 SD) - {movie}",
            out_path=specific_dir,
            name=f"movie_specific_small_network_effects_1SD_{movie_safe}",
        )

    print("Saved outputs to:", out_dir)
    print("Saved region-to-network lookup to:", os.path.join(out_dir, "region_to_network_lookup.csv"))
    print("Saved network x movie matrix to:", os.path.join(out_dir, "network_by_movie_score_matrix.csv"))

    print("Saved static 3D brains to:", os.path.join(out_dir, "3d_brains"))
    print("Saved interactive 3D brains to:", os.path.join(out_dir, "3d_brains_html"))
    print("Saved value images to:", os.path.join(out_dir, "value_images"))

    print("Saved glass brains to:", os.path.join(out_dir, "glassbrains"))
    print("Saved slices to:", os.path.join(out_dir, "slices"))

    print("Saved movie-specific network outputs to:", specific_dir)
    print("Saved static movie-specific 3D brains to:", os.path.join(specific_dir, "3d_brains"))
    print("Saved interactive movie-specific 3D brains to:", os.path.join(specific_dir, "3d_brains_html"))
    print("Saved movie-specific value images to:", os.path.join(specific_dir, "value_images"))

    print("Saved movie-specific glass brains to:", os.path.join(specific_dir, "glassbrains"))
    print("Saved movie-specific slices to:", os.path.join(specific_dir, "slices"))