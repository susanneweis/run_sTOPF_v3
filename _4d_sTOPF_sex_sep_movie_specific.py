import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import os


def main(base_path, proj, code):
    # -----------------------
    # Paths
    # -----------------------
    results_path = f"{base_path}/results_run_sTOPF_{code}_data_{proj}"
    res_in_dir = f"{results_path}/sex_separability"

    infile = f"{res_in_dir}/ALL_movies_sex_diff_scores.csv"
    outdir = f"{res_in_dir}/movie_specific_sex_separation"
    os.makedirs(outdir, exist_ok=True)

    exclude_movies = {"REST1", "REST2", "concat"}

    # =============================
    # Load data
    # =============================

    df = pd.read_csv(infile)

    # Keep only relevant columns
    df = df[["Region", "score", "movie"]].copy()

    # Exclude rest and concat
    df = df[~df["movie"].astype(str).isin(exclude_movies)].copy()

    # =============================
    # Extract Yeo-17 network
    # =============================

    def extract_network(region):
        region = str(region)

        # Example:
        # 17Networks_LH_DefaultA_...
        # 17Networks_RH_ContB_...
        m = re.match(r"17Networks_[LR]H_([^_]+)_", region)

        if m:
            return m.group(1)
        else:
            return "subcortical"

    df["Network"] = df["Region"].apply(extract_network)

    # =============================
    # Aggregate regions to networks
    # =============================

    network_movie = (
        df.groupby(["movie", "Network"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            sd_score=("score", "std"),
            n_regions=("Region", "nunique")
        )
    )

    network_movie.to_csv(f"{outdir}/sex_separation_network_by_movie_summary.csv", index=False)

    # =============================
    # Create matrices
    # =============================

    # Sort networks by overall mean sex-separation score
    network_order = (
        network_movie.groupby("Network")["mean_score"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # Keep movies in original file order
    movie_order = df["movie"].drop_duplicates().tolist()

    # Absolute score matrix
    abs_mat = (
        network_movie
        .pivot(index="Network", columns="movie", values="mean_score")
        .reindex(index=network_order, columns=movie_order)
    )

    # Relative movie-specific deviation matrix
    # z-score each network across movies
    rel_mat = abs_mat.sub(abs_mat.mean(axis=1), axis=0)
    rel_mat = rel_mat.div(abs_mat.std(axis=1).replace(0, np.nan), axis=0)

    abs_mat.to_csv(f"{outdir}/absolute_network_movie_matrix.csv")
    rel_mat.to_csv(f"{outdir}/relative_network_movie_z_matrix.csv")

    # =============================
    # Plot
    # =============================

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 9),
        constrained_layout=True
    )

    # ---- Left: absolute scores ----
    im0 = axes[0].imshow(
        abs_mat.values,
        aspect="auto",
        cmap="hot"
    )

    axes[0].set_title(
        "Absolute sex-separation scores\n(high = stronger sex separation)",
        fontsize=14,
        weight="bold"
    )

    # ---- Right: relative deviations ----
    vmax = np.nanmax(np.abs(rel_mat.values))

    im1 = axes[1].imshow(
        rel_mat.values,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax
    )

    axes[1].set_title(
        "Relative movie-specific deviations\n(red = unusually high, blue = unusually low)",
        fontsize=14,
        weight="bold"
    )

    # ---- Axis formatting ----
    for ax, mat in zip(axes, [abs_mat, rel_mat]):
        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right")

        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index)

        ax.set_xlabel("Movie")
        ax.set_ylabel("Network")

        ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
        ax.grid(which="minor", linewidth=0.4)
        ax.tick_params(which="minor", bottom=False, left=False)

    # ---- Add numbers to cells ----
    for i in range(abs_mat.shape[0]):
        for j in range(abs_mat.shape[1]):

            abs_val = abs_mat.iloc[i, j]
            rel_val = rel_mat.iloc[i, j]

            if np.isfinite(abs_val):
                axes[0].text(
                    j, i, f"{abs_val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs_val < abs_mat.max().max() * 0.65 else "black"
                )

            if np.isfinite(rel_val):
                axes[1].text(
                    j, i, f"{rel_val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7
                )

    # ---- Colorbars ----
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Mean sex-separation score")

    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Within-network z-score across movies")

    fig.suptitle(
        "Sex separation across movies and functional networks",
        fontsize=18,
        weight="bold"
    )

    # Save figure
    fig.savefig(f"{outdir}/sex_separation_network_movie_heatmaps.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{outdir}/sex_separation_network_movie_heatmaps.pdf", bbox_inches="tight")

    plt.show()

    # =============================
    # Identify striking boxes
    # =============================

    # Most unusually high cells
    high_cells = (
        rel_mat.stack()
        .reset_index()
        .rename(columns={0: "relative_z"})
        .sort_values("relative_z", ascending=False)
    )

    # Most unusually low cells
    low_cells = (
        rel_mat.stack()
        .reset_index()
        .rename(columns={0: "relative_z"})
        .sort_values("relative_z", ascending=True)
    )

    high_cells["absolute_mean_score"] = high_cells.apply(
        lambda row: abs_mat.loc[row["Network"], row["movie"]],
        axis=1
    )

    low_cells["absolute_mean_score"] = low_cells.apply(
        lambda row: abs_mat.loc[row["Network"], row["movie"]],
        axis=1
    )

    high_cells.to_csv(f"{outdir}/most_unusually_high_network_movie_cells.csv", index=False)
    low_cells.to_csv(f"{outdir}/most_unusually_low_network_movie_cells.csv", index=False)

    print("\nMost unusually high movie x network cells:")
    print(high_cells.head(10))

    print("\nMost unusually low movie x network cells:")
    print(low_cells.head(10))

    print(f"\nSaved all outputs to: {outdir}")