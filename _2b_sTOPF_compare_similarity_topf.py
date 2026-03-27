
import pandas as pd
import seaborn as sns


def compute_region_summary(pairwise_file, explained_var_file, movie_name, value_col="value"):
    """
    pairwise_file: long file (region, subject_1, subject_2, value)
    explained_var_file: region-wise explained variance file
    """

    # -------------------------
    # load data
    # -------------------------
    df_pair = pd.read_csv(pairwise_file)
    df_exp = pd.read_csv(explained_var_file)

    # -------------------------
    # 1. mean similarity per region
    # -------------------------
    df_mean = (
        df_pair
        .groupby("region")[value_col]
        .mean()
        .reset_index()
        .rename(columns={value_col: "mean_similarity"})
    )

    # -------------------------
    # 2. explained variance
    # adjust column name if needed
    # -------------------------
    # assume columns: region + something like "explained_variance"
    exp_col = [c for c in df_exp.columns if "var" in c.lower()][0]

    df_exp = df_exp.rename(columns={exp_col: "explained_variance"})
    df_exp = df_exp.rename(columns={"Region": "region"})

    # -------------------------
    # 3. merge
    # -------------------------
    df_merge = pd.merge(df_mean, df_exp, on="region", how="inner")
    df_merge["movie"] = movie_name

    return df_merge

import glob
import os

def build_full_dataset(pairwise_dir, pair_file, explained_var_dir, expl_file, metric):
    """
    metric: 'corr' or 'mi'
    """

    all_dfs = []

    movie_dirs = sorted([
        d for d in glob.glob(os.path.join(pairwise_dir, "*"))
        if os.path.isdir(d)
    ])

    for mov_dir in movie_dirs:
        movie = os.path.basename(mov_dir)

        pairwise_file = os.path.join(
            mov_dir, f"{movie}_{pair_file}_{metric}_long.csv"
        )

        explained_file = os.path.join(
            explained_var_dir,
            f"{movie}/{expl_file}.csv"  # adjust if needed
        )

        if not os.path.exists(pairwise_file):
            print(f"Missing pairwise file for {movie}")
            continue

        if not os.path.exists(explained_file):
            print(f"Missing explained variance file for {movie}")
            continue

        df = compute_region_summary(pairwise_file, explained_file, movie)
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    return full_df


import matplotlib.pyplot as plt


def plot_similarity_vs_variance(full_df, title, out_path, out_file):
    
    plt.figure(figsize=(6, 5))

    plt.scatter(
        full_df["explained_variance"],
        full_df["mean_similarity"],
        alpha=0.5
    )

    plt.xlabel("Explained variance (PC1)")
    plt.ylabel("Mean intersubject similarity")
    plt.title(title)

    # optional: correlation line
    r = full_df["explained_variance"].corr(full_df["mean_similarity"])
    plt.title(f"{title} (r = {r:.2f})")

    plt.tight_layout()
    plt.show()

    plt.savefig(f"{out_path}/{out_file}", bbox_inches='tight',dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))

    sns.scatterplot(
        data=full_df,
        x="explained_variance",
        y="mean_similarity",
        hue="movie",
        alpha=0.4
    )

    plt.xlabel("Explained variance (PC1)")
    plt.ylabel("Mean similarity")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(f"{out_path}/{out_file}_2", dpi=300)
    plt.close()


def main(base_path,proj,code,movies_properties,nn): 

    pairwise_dir = f"{base_path}/results_run_sTOPF_{code}_data_{proj}/pairwise_subject_similarity_nn{nn}/"

    #movies = list(movies_properties.keys())

    for metric in ["corr", f"mi_nn{nn}"]:
        # all
        pair_file = "all"
        explained_var_dir = f"{base_path}/results_run_sTOPF_{code}_data_{proj}/results_PCA_all/"
        explained_file = f"explained_variance_1_{pair_file}_allROI"
        title= f"Correlation vs Explained Variance {pair_file} {metric}"
        out_file = f"Corr_vs_ExpVar_{pair_file}_{metric}"

        df_corr = build_full_dataset(pairwise_dir, pair_file, explained_var_dir, explained_file, metric)
        plot_similarity_vs_variance(df_corr, title, pairwise_dir, out_file)

        # female
        pair_file = "female"
        explained_var_dir = f"{base_path}/results_run_sTOPF_{code}_data_{proj}/results_PCA_per_sex/"
        explained_file = f"explained_variance_1_{pair_file}_allROI"
        title= f"Correlation vs Explained Variance {pair_file} {metric}"
        out_file = f"Corr_vs_ExpVar_{pair_file}_{metric}"

        df_corr = build_full_dataset(pairwise_dir, pair_file, explained_var_dir, explained_file, metric)
        plot_similarity_vs_variance(df_corr, title, pairwise_dir, out_file)

        # male
        pair_file = "male"
        explained_var_dir = f"{base_path}/results_run_sTOPF_{code}_data_{proj}/results_PCA_per_sex/"
        explained_file = f"explained_variance_1_{pair_file}_allROI"
        title= f"Correlation vs Explained Variance {pair_file} {metric}"
        out_file = f"Corr_vs_ExpVar_{pair_file}_{metric}"

        df_corr = build_full_dataset(pairwise_dir, pair_file, explained_var_dir, explained_file, metric)
        plot_similarity_vs_variance(df_corr, title, pairwise_dir, out_file)

    