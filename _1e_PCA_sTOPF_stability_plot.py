import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def plot_scatter_per_movie(df, out_dir):
    """
    Scatter plot for each movie:
    x = mean_stability_female
    y = mean_stability_male
    """
    movies = sorted(df["movie"].dropna().unique())

    for movie in movies:
        sub = df[df["movie"] == movie].copy()

        plt.figure(figsize=(6, 6))
        plt.scatter(
            sub["mean_stability_female"],
            sub["mean_stability_male"],
            alpha=0.6
        )

        min_val = np.nanmin([
            sub["mean_stability_female"].min(),
            sub["mean_stability_male"].min()
        ])
        max_val = np.nanmax([
            sub["mean_stability_female"].max(),
            sub["mean_stability_male"].max()
        ])

        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        plt.xlabel("Mean stability female")
        plt.ylabel("Mean stability male")
        plt.title(f"Female vs male stability: {movie}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"scatter_female_vs_male_{movie}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


def plot_scatter_average(df, out_dir):
    """
    Scatter plot after averaging across movies for each region.
    """
    avg_df = (
        df.groupby("region", as_index=False)[
            ["mean_stability_female", "mean_stability_male"]
        ]
        .mean()
    )

    plt.figure(figsize=(6, 6))
    plt.scatter(
        avg_df["mean_stability_female"],
        avg_df["mean_stability_male"],
        alpha=0.6
    )

    min_val = np.nanmin([
        avg_df["mean_stability_female"].min(),
        avg_df["mean_stability_male"].min()
    ])
    max_val = np.nanmax([
        avg_df["mean_stability_female"].max(),
        avg_df["mean_stability_male"].max()
    ])

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Mean stability female")
    plt.ylabel("Mean stability male")
    plt.title("Female vs male stability (averaged across movies)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "scatter_female_vs_male_avg_across_movies.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_heatmap(matrix, title, out_file, figsize=(12, 10)):
    """
    Simple heatmap using matplotlib only.
    Rows = regions
    Columns = movies
    """
    plt.figure(figsize=figsize)
    im = plt.imshow(matrix.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="Stability")

    plt.xticks(
        ticks=np.arange(len(matrix.columns)),
        labels=matrix.columns,
        rotation=90
    )
    plt.yticks(
        ticks=np.arange(len(matrix.index)),
        labels=matrix.index
    )

    plt.title(title)
    plt.xlabel("Movie")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmaps(df, out_dir):
    """
    Create heatmaps:
    1) female stability
    2) male stability
    3) female - male difference
    """
    female_mat = df.pivot(index="region", columns="movie", values="mean_stability_female")
    male_mat = df.pivot(index="region", columns="movie", values="mean_stability_male")
    diff_mat = female_mat - male_mat

    plot_heatmap(
        female_mat,
        title="Female PCA stability across movies",
        out_file=os.path.join(out_dir, "heatmap_stability_female.png")
    )

    plot_heatmap(
        male_mat,
        title="Male PCA stability across movies",
        out_file=os.path.join(out_dir, "heatmap_stability_male.png")
    )

    plot_heatmap(
        diff_mat,
        title="Female - male PCA stability across movies",
        out_file=os.path.join(out_dir, "heatmap_stability_female_minus_male.png")
    )


def plot_distributions(df, out_dir):
    """
    Distribution plots of stability values across all movie-region entries.
    """
    female_vals = df["mean_stability_female"].dropna()
    male_vals = df["mean_stability_male"].dropna()

    plt.figure(figsize=(7, 5))
    plt.hist(female_vals, bins=30, alpha=0.5, label="Female")
    plt.hist(male_vals, bins=30, alpha=0.5, label="Male")
    plt.xlabel("Mean stability")
    plt.ylabel("Count")
    plt.title("Distribution of PCA stability values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "distribution_stability_female_male.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_distributions_average(df, out_dir):
    """
    Distribution plot after averaging over movies within region.
    """
    avg_df = (
        df.groupby("region", as_index=False)[
            ["mean_stability_female", "mean_stability_male"]
        ]
        .mean()
    )

    female_vals = avg_df["mean_stability_female"].dropna()
    male_vals = avg_df["mean_stability_male"].dropna()

    plt.figure(figsize=(7, 5))
    plt.hist(female_vals, bins=30, alpha=0.5, label="Female")
    plt.hist(male_vals, bins=30, alpha=0.5, label="Male")
    plt.xlabel("Mean stability")
    plt.ylabel("Count")
    plt.title("Distribution of PCA stability values (averaged across movies)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "distribution_stability_female_male_avg_across_movies.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def main(base_path, proj, code):
    in_dir  = f"{base_path}/results_run_sTOPF_{code}_data_{proj}"
    infile = os.path.join(in_dir, "PC1_stability_summary.csv")

    out_dir = os.path.join(in_dir, "results_PC1_stability_visualizations")
    os.makedirs(out_dir, exist_ok=True)


    df = pd.read_csv(infile)

    # Scatter plots
    plot_scatter_per_movie(df, out_dir)
    plot_scatter_average(df, out_dir)

    # Heatmaps
    plot_heatmaps(df, out_dir)

    # Distribution plots
    plot_distributions(df, out_dir)
    plot_distributions_average(df, out_dir)

    print(f"Saved all plots to: {out_dir}")


if __name__ == "__main__":
    main()