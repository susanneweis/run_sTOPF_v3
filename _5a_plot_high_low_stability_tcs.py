import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def main(base_path, proj, code, nn, top_n):
    # =========================
    # Paths
    # =========================
    pca_base_dir = f"{base_path}/results_run_sTOPF_{code}_data_{proj}/results_PCA_per_sex"
    stability_file = f"{base_path}/results_run_sTOPF_{code}_data_{proj}/results_nn{nn}/sexability_nn{nn}/sexability_sensitivity_no_sub_fac_with_ss_nn{nn}/shared_map_all_movies_fem_male_diff_corr.csv"

    output_dir = f"{base_path}/results_run_sTOPF_{code}_data_{proj}/results_nn{nn}/stability_tc_plots_nn{nn}/"
    os.makedirs(output_dir, exist_ok=True)


    # =========================
    # Load stability file
    # =========================
    shared_df = pd.read_csv(stability_file)

    #top_n = 10

    lowest_df = (
        shared_df.sort_values("stability_score", ascending=True)
        .head(top_n)
        .copy()
    )

    highest_df = (
        shared_df.sort_values("stability_score", ascending=False)
        .head(top_n)
        .copy()
    )

    lowest_regions = lowest_df["region"].tolist()
    highest_regions = highest_df["region"].tolist()


    # =========================
    # Save region CSV files
    # =========================
    lowest_csv = os.path.join(output_dir, f"lowest_{top_n}_stability_regions.csv")
    highest_csv = os.path.join(output_dir, f"highest_{top_n}_stability_regions.csv")

    lowest_df.to_csv(lowest_csv, index=False)
    highest_df.to_csv(highest_csv, index=False)


    # =========================
    # Plotting function
    # =========================
    def plot_region_set(male_df, female_df, region_list, movie_name, set_name, out_file):
        fig, axes = plt.subplots(
            len(region_list),
            1,
            figsize=(12, 2.5 * len(region_list)),
            sharex=False
        )

        if len(region_list) == 1:
            axes = [axes]

        # common y-axis within figure
        all_vals = []
        for region in region_list:
            all_vals.extend(male_df.loc[male_df["Region"] == region, "PC_score_1"].tolist())
            all_vals.extend(female_df.loc[female_df["Region"] == region, "PC_score_1"].tolist())

        if len(all_vals) > 0:
            y_min = min(all_vals)
            y_max = max(all_vals)
        else:
            y_min, y_max = -1, 1

        for ax, region in zip(axes, region_list):
            male_ts = male_df.loc[male_df["Region"] == region, "PC_score_1"].values
            female_ts = female_df.loc[female_df["Region"] == region, "PC_score_1"].values

            if len(male_ts) == 0 and len(female_ts) == 0:
                ax.text(
                    0.5, 0.5,
                    f"{region}\nnot found",
                    ha="center", va="center",
                    transform=ax.transAxes
                )
                ax.set_title(region)
                ax.set_ylim(y_min, y_max)
                continue

            if len(male_ts) > 0:
                ax.plot(male_ts, label="Male")
            if len(female_ts) > 0:
                ax.plot(female_ts, label="Female")

            ax.set_title(region)
            ax.set_ylim(y_min, y_max)
            ax.legend()

        fig.suptitle(f"{movie_name} – {set_name}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()


    # =========================
    # Loop over movies
    # =========================
    movie_dirs = sorted(
        [d for d in glob.glob(os.path.join(pca_base_dir, "*")) if os.path.isdir(d)]
    )

    for movie_dir in movie_dirs:
        movie_name = os.path.basename(movie_dir)

        male_file = os.path.join(movie_dir, "PC1_scores_male_allROI.csv")
        female_file = os.path.join(movie_dir, "PC1_scores_female_allROI.csv")

        if not os.path.exists(male_file) or not os.path.exists(female_file):
            print(f"Skipping {movie_name}: missing male or female PC1 file.")
            continue

        print(f"Processing {movie_name}")

        male_df = pd.read_csv(male_file)
        female_df = pd.read_csv(female_file)

        out_low = os.path.join(
            output_dir,
            f"{movie_name}_lowest_{top_n}_stability_regions_PC1.png"
        )
        out_high = os.path.join(
            output_dir,
            f"{movie_name}_highest_{top_n}_stability_regions_PC1.png"
        )

        plot_region_set(
            male_df=male_df,
            female_df=female_df,
            region_list=lowest_regions,
            movie_name=movie_name,
            set_name=f"lowest {top_n} stability regions",
            out_file=out_low
        )

        plot_region_set(
            male_df=male_df,
            female_df=female_df,
            region_list=highest_regions,
            movie_name=movie_name,
            set_name=f"highest {top_n} stability regions",
            out_file=out_high
        )

    print("Done.")
    print(f"Plots saved to: {output_dir}")
    print(f"Lowest stability CSV: {lowest_csv}")
    print(f"Highest stability CSV: {highest_csv}")


# Execute script
if __name__ == "__main__":
    main()
