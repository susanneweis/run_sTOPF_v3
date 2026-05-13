import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

network_names = [
    "VisCent", "VisPeri",
    "SomMotA", "SomMotB",
    "DorsAttnA", "DorsAttnB",
    "SalVentAttnA", "SalVentAttnB",
    "LimbicA", "LimbicB",
    "ContA", "ContB", "ContC",
    "DefaultA", "DefaultB", "DefaultC",
    "TempPar",
]

def extract_network(region):
    region = str(region)
    for net in network_names:
        if net in region:
            return net
    return "Subcortical"


def main(base_path, proj, code):
    # -----------------------
    # Paths
    # -----------------------

    results_path = f"{base_path}/results_run_sTOPF_{code}_data_{proj}"
    res_in_dir = f"{results_path}/sex_separability"

    infile = f"{res_in_dir}/ALL_movies_sex_diff_scores.csv"

    outdir = f"{res_in_dir}/sex_sep_vs_unexplained_variance"
    os.makedirs(outdir, exist_ok=True)

    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_csv(infile)

    # -----------------------
    # Extract network
    # -----------------------


    df["Network"] = df["Region"].apply(extract_network)

    # -----------------------
    # Compute mean unexplained variance
    # -----------------------
    df["mean_unexplained_variance"] = (
        (1 - df["EV_female"]) + (1 - df["EV_male"])
    ) / 2

    # Optional: save enriched table
    df.to_csv(
        f"{outdir}/ALL_movies_score_mean_unexplained_variance_with_network.csv",
        index=False,
    )

    # -----------------------
    # Plot one scatter plot per movie
    # -----------------------
    for movie, dmovie in df.groupby("movie", sort=True):

        fig, ax = plt.subplots(figsize=(8, 6))

        for network, dnet in dmovie.groupby("Network", sort=True):
            ax.scatter(
                dnet["mean_unexplained_variance"],
                dnet["score"],
                label=network,
                alpha=0.75,
                s=35,
                edgecolors="none",
            )

        ax.set_title(f"{movie}: score vs mean unexplained variance")
        ax.set_xlabel("Mean unexplained variance: mean(1 - EV female, 1 - EV male)")
        ax.set_ylabel("Score")

        ax.legend(
            title="Network",
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=8,
        )

        fig.tight_layout()

        outfile = f"{outdir}/{movie}_score_vs_mean_unexplained_variance_by_network.png"
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved plots to: {outdir}")