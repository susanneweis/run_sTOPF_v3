import os
import numpy as np
import pandas as pd
from _util_glass_brains_borders import create_glassbrains


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_least_stable_regions_heatmap(shared_map_file, movies, res_path, top_n, output_file, group, met):
    """
    Plot heatmap of normalized residuals across movies for the least stable regions.

    Parameters
    ----------
    shared_map_file : str or Path
        CSV file with shared-map summary. Must contain:
        - 'region'
        - 'stability_score'

    movie_specific_dir : str or Path
        Directory containing files like:
        movie_specific_DD_corr.csv
        movie_specific_S_corr.csv
        ...

    top_n : int
        Number of least stable regions to include.

    use_abs : bool
        If True, use 'abs_normalized_residual'.
        If False, use signed 'normalized_residual'.

    output_file : str or Path or None
        If given, save figure to this file.
    """
    # -----------------------------
    # 1. Load shared map
    # -----------------------------
    shared_df = pd.read_csv(shared_map_file)

    required_shared_cols = {"region", "stability_score"}
    missing_shared = required_shared_cols - set(shared_df.columns)
    if missing_shared:
        raise ValueError(
            f"Shared map file is missing columns: {missing_shared}\n"
            f"Available columns: {list(shared_df.columns)}"
        )

    # -----------------------------
    # 2. Select least stable regions
    # -----------------------------
    least_stable_df = (
        shared_df.sort_values("stability_score", ascending=True)
                 .head(top_n)
                 .copy()
    )

    least_stable_regions = least_stable_df["region"].tolist()

    # -----------------------------
    # 3. Load all movie-specific files
    # -----------------------------

    residual_col = "normalized_residual"

    movie_dfs = []
    for mv in movies:
        mv_f = f"{res_path}/movie_specific_{mv}_{group}_{met}.csv"
        df = pd.read_csv(mv_f)

        required_movie_cols = {"region", residual_col}
        missing_movie = required_movie_cols - set(df.columns)
        if missing_movie:
            raise ValueError(
                f"Movie file {mv} is missing columns: {missing_movie}\n"
                f"Available columns: {list(df.columns)}"
            )

        tmp = df.loc[df["region"].isin(least_stable_regions), ["region", residual_col]].copy()
        tmp["movie"] = mv
        movie_dfs.append(tmp)

    all_movies_df = pd.concat(movie_dfs, ignore_index=True)

    # -----------------------------
    # 4. Build region x movie matrix
    # -----------------------------
    heat_df = all_movies_df.pivot(
        index="region",
        columns="movie",
        values=residual_col
    )

    # keep row order from least to less least stable
    heat_df = heat_df.loc[least_stable_regions]

    # optional: keep movies in filename order
    heat_df = heat_df.reindex(sorted(heat_df.columns), axis=1)

    # -----------------------------
    # 5. Plot heatmap
    # -----------------------------
    plt.figure(figsize=(1.2 * max(4, heat_df.shape[1]), 0.35 * max(8, heat_df.shape[0])))

  
    vmax = pd.Series(heat_df.values.ravel()).abs().max()
    im = plt.imshow(heat_df.values, aspect="auto", vmin=-vmax, vmax=vmax)
    cbar_label = "Normalized residual"

    plt.colorbar(im, label=cbar_label)
    plt.xticks(range(len(heat_df.columns)), heat_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(heat_df.index)), heat_df.index)
    plt.title(f"Least stable regions (top {top_n}) across movies")
    plt.xlabel("Movie")
    plt.ylabel("Region")
    plt.tight_layout()

    plt.savefig(f"{res_path}/{output_file}.png", dpi=300, bbox_inches="tight")

    plt.show()

def plot_summary_heatmap(summary_file, out_path, out_file):
    # Load
    df = pd.read_csv(summary_file)

    # Keep only numeric columns
    df = df.select_dtypes(include="number")

    # Split
    abs_cols = [c for c in df.columns if "abs" in c.lower()]
    other_cols = [c for c in df.columns if c not in abs_cols]

    df_abs = df[abs_cols]
    df_other = df[other_cols]

    # ---------
    # PLOTTING
    # ---------
    def plot_heatmap(data, title, out_path):
        plt.figure(figsize=(1.2 * data.shape[1], 0.4 * data.shape[0]))
        plt.imshow(data.values, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(data.columns)), data.columns, rotation=45, ha="right")
        plt.yticks(range(len(data.index)), data.index)
        plt.title(title)
        plt.tight_layout()
        plt.show()

        plt.savefig(out_path, dpi=300)
        plt.close()  # important to avoid overlapping figures

    # Plot both
    out_f_abs = f"{out_path}/{out_file}_abs.png"
    plot_heatmap(df_abs, "Abs columns",out_f_abs)

    out_f = f"{out_path}/{out_file}_abs.png"
    plot_heatmap(df_other, "Other columns",out_f)


def main(base_path, proj, code, nn_mi, mov_prop, atlas_path, roi_names):

    """
    Computes:
    1) shared map across all movies (mean per region)
    2) leave-one-movie-out mean map for each movie
    3) movie-specific residual map: movie - mean(other movies)
    4) normalized residual map: (movie - mean(other movies)) / sd(other movies)
    5) ranked tables for most shared / most variable / most movie-specific regions
    """
    movies = list(mov_prop.keys())
    # actual movies
    movies = movies[:-2]   # excluding last two, as in your script

    results_path = f"{base_path}/results_run_sTOPF_{code}_data_{proj}/results_nn{nn_mi}"    
    in_path = f"{results_path}/sexability_nn{nn_mi}"

    if len(movies) == 7:
        results_out = f"{in_path}/shared_and_specific_nn{nn_mi}"
    elif len(movies) == 8:
        results_out = f"{in_path}/shared_and_specific_ss_nn{nn_mi}"

    region_col="region"
    score = "cohens_d"
    eps=1e-8
    #top_n=max_regions

    # ------------------------------------------------------------------
    # 1. Read all movie files and collect the chosen metric
    # ------------------------------------------------------------------

    for group in ["female", "male", "fem_male_diff"]:

        for met in ["corr", f"mi_nn{nn_mi}"]:

            results_out_path = f"{results_out}/{met}"
            os.makedirs(results_out_path, exist_ok=True)

            results_g_path = f"{results_out_path}/figures"
            os.makedirs(results_g_path, exist_ok=True)

            dfs = []

            for movie in movies:

                file_path = f"{in_path}/{met}/cohens_d_{movie}_{met}_{group}.csv"

                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found for movie {movie}: {file_path}")

                df = pd.read_csv(file_path)

                if region_col not in df.columns:
                    raise ValueError(f"Column '{region_col}' not found in {file_path}")

                if score not in df.columns:
                    raise ValueError(f"Column '{score}' not found in {file_path}")

                curr = df[[region_col, score]].copy()
                curr = curr.rename(columns={score: movie})
                dfs.append(curr)

            # ------------------------------------------------------------------
            # 2. Merge into one region x movie matrix
            # ------------------------------------------------------------------
            data_wide = dfs[0]
            for curr in dfs[1:]:
                data_wide = data_wide.merge(curr, on=region_col, how="outer")

            data_wide = data_wide.sort_values(region_col).reset_index(drop=True)
            movie_cols = list(movies)

            data_wide.to_csv(
                os.path.join(results_out_path, f"region_by_movie_{group}_{met}.csv"),
                index=False
            )

            # ------------------------------------------------------------------
            # 3. Shared map across all movies
            # ------------------------------------------------------------------
            shared_map = data_wide[[region_col]].copy()
            shared_map["mean_all_movies"] = data_wide[movie_cols].mean(axis=1)
            shared_map["std_all_movies"] = data_wide[movie_cols].std(axis=1, ddof=1)
            shared_map["var_all_movies"] = data_wide[movie_cols].var(axis=1, ddof=1)
            shared_map["abs_mean_all_movies"] = shared_map["mean_all_movies"].abs()

            # strong and stable
            shared_map["stability_score"] = (
                shared_map["abs_mean_all_movies"] / (shared_map["std_all_movies"] + eps)
            )

            shared_map_file = f"{results_out_path}/shared_map_all_movies_{group}_{met}.csv"
            shared_map.to_csv(shared_map_file, index=False)

            create_glassbrains(
                shared_map_file,
                "stability_score",
                "region",
                roi_names,
                atlas_path,
                f"All movies: stability score {group} {met}",
                results_g_path,
                f"Stability_all_movies_{group}_{met}",
                "continuous", 
                shared_map["stability_score"].min(), 
                shared_map["stability_score"].max()
            )


            # ------------------------------------------------------------------
            # 4. Ranked tables across all movies
            # ------------------------------------------------------------------
            # Most shared = high absolute mean + low variability
            # most_shared = shared_map.sort_values(
            #     by="stability_score", ascending=False
            # ).reset_index(drop=True)

            # most_shared.to_csv(
            #     os.path.join(results_out_path, f"ranked_most_shared_regions_{group}_{met}.csv"),
            #     index=False
            # )

            # most_shared.head(top_n).to_csv(
            #     os.path.join(results_out_path, f"top_{top_n}_most_shared_regions_{metric}.csv"),
            #     index=False
            # )

            # Most variable across movies
            # most_variable = shared_map.sort_values(
            #     by="std_all_movies", ascending=False
            # ).reset_index(drop=True)

            # most_variable.to_csv(
            #     os.path.join(results_out_path, f"ranked_most_variable_regions_{group}_{met}.csv"),
            #     index=False
            # )

            # most_variable.head(top_n).to_csv(
            #     os.path.join(results_out_path, f"top_{top_n}_most_variable_regions_{metric}.csv"),
            #     index=False
            # )

            # Strongest mean effects regardless of stability
            # strongest_mean = shared_map.sort_values(
            #     by="abs_mean_all_movies", ascending=False
            # ).reset_index(drop=True)

            # strongest_mean.to_csv(
            #     os.path.join(results_out_path, f"ranked_strongest_mean_regions_{group}_{met}.csv"),
            #     index=False
            # )

            # strongest_mean.head(top_n).to_csv(
            #     os.path.join(results_out_path, f"top_{top_n}_strongest_mean_regions_{metric}.csv"),
            #     index=False
            # )

            # ------------------------------------------------------------------
            # 5. Per-movie leave-one-out residuals and ranked outputs
            # ------------------------------------------------------------------
            summary_rows = []

            for movie in movie_cols:
                other_movies = [m for m in movie_cols if m != movie]

                out_df = data_wide[[region_col]].copy()
                out_df[f"{movie}_score"] = data_wide[movie]
                out_df["mean_other_movies"] = data_wide[other_movies].mean(axis=1)
                out_df["std_other_movies"] = data_wide[other_movies].std(axis=1, ddof=1)
                out_df["var_other_movies"] = data_wide[other_movies].var(axis=1, ddof=1)

                out_df["residual"] = out_df[f"{movie}_score"] - out_df["mean_other_movies"]
                out_df["normalized_residual"] = (
                    out_df["residual"] / (out_df["std_other_movies"] + eps)
                )

                out_df["abs_residual"] = out_df["residual"].abs()
                out_df["abs_normalized_residual"] = out_df["normalized_residual"].abs()

                spec_out_file = f"{results_out_path}/movie_specific_{movie}_{group}_{met}.csv"
                # Save full per-movie table
                out_df.to_csv(spec_out_file,index=False)

                create_glassbrains(
                    spec_out_file,
                    "normalized_residual",
                    "region",
                    roi_names,
                    atlas_path,
                    f"{movie} {group} {met}: Normalized residual",
                    results_g_path,
                    f"Normalized_residual_{movie}_{group}_{met}",
                    "continuous",
                    -10,
                    10
                )


                # -----------------------------
                # Ranked residual tables
                # -----------------------------
                # Positive residuals = stronger than expected in this movie
                # pos_resid = out_df.sort_values(by="residual", ascending=False).reset_index(drop=True)
                # pos_resid.to_csv(
                #     os.path.join(results_out_path, f"ranked_positive_residuals_{movie}_{group}_{met}.csv"),
                #     index=False
                # )
                # pos_resid.head(top_n).to_csv(
                #     os.path.join(results_out_path, f"top_{top_n}_positive_residuals_{movie}_{metric}.csv"),
                #     index=False
                # )

                # Negative residuals = weaker than expected in this movie
                # neg_resid = out_df.sort_values(by="residual", ascending=True).reset_index(drop=True)
                # neg_resid.to_csv(
                #     os.path.join(results_out_path, f"ranked_negative_residuals_{movie}_{group}_{met}.csv"),
                #     index=False
                # )
                # neg_resid.head(top_n).to_csv(
                #     os.path.join(results_out_path, f"top_{top_n}_negative_residuals_{movie}_{metric}.csv"),
                #     index=False
                # )

                # Absolute residuals = strongest deviations regardless of direction
                # abs_resid = out_df.sort_values(by="abs_residual", ascending=False).reset_index(drop=True)
                # abs_resid.to_csv(
                #     os.path.join(results_out_path, f"ranked_absolute_residuals_{movie}_{group}_{met}.csv"),
                #     index=False
                # )
                # abs_resid.head(top_n).to_csv(
                #     os.path.join(results_out_path, f"top_{top_n}_absolute_residuals_{movie}_{metric}.csv"),
                #     index=False
                # )

                # Positive normalized residuals
                # pos_norm = out_df.sort_values(by="normalized_residual", ascending=False).reset_index(drop=True)
                # pos_norm.to_csv(
                #     os.path.join(results_out_path, f"ranked_positive_normalized_residuals_{movie}_{group}_{met}.csv"),
                #     index=False
                # )
                # pos_norm.head(top_n).to_csv(
                #     os.path.join(results_out_path, f"top_{top_n}_positive_normalized_residuals_{movie}_{metric}.csv"),
                #     index=False
                # )

                # Negative normalized residuals
                # neg_norm = out_df.sort_values(by="normalized_residual", ascending=True).reset_index(drop=True)
                # neg_norm.to_csv(
                #     os.path.join(results_out_path, f"ranked_negative_normalized_residuals_{movie}_{group}_{met}.csv"),
                #     index=False
                # )
                # neg_norm.head(top_n).to_csv(
                #     os.path.join(results_out_path, f"top_{top_n}_negative_normalized_residuals_{movie}_{metric}.csv"),
                #     index=False
                # )

                # Absolute normalized residuals
                # abs_norm = out_df.sort_values(by="abs_normalized_residual", ascending=False).reset_index(drop=True)
                # abs_norm.to_csv(
                #     os.path.join(results_out_path, f"ranked_absolute_normalized_residuals_{movie}_{group}_{met}.csv"),
                #     index=False
                # )
                # abs_norm.head(top_n).to_csv(
                #     os.path.join(results_out_path, f"top_{top_n}_absolute_normalized_residuals_{movie}_{metric}.csv"),
                #     index=False
                # )

                # Movie-level summary
                summary_rows.append({
                    "movie": movie,
                    "mean_residual": out_df["residual"].mean(),
                    "max_residual": out_df["residual"].max(),
                    "min_residual": out_df["residual"].min(),
                    "mean_normalized_residual": out_df["normalized_residual"].mean(),
                    "max_normalized_residual": out_df["normalized_residual"].max(),
                    "min_normalized_residual": out_df["normalized_residual"].min(),
                    "mean_abs_residual": out_df["abs_residual"].mean(),
                    "max_abs_residual": out_df["abs_residual"].max(),
                    "mean_abs_normalized_residual": out_df["abs_normalized_residual"].mean(),
                    "max_abs_normalized_residual": out_df["abs_normalized_residual"].max(),
                })

            summary_df = pd.DataFrame(summary_rows)
            summary_file = f"{results_out_path}/movie_specific_summary_{group}_{met}.csv"
            summary_df.to_csv(summary_file,index=False)

            print("Done.")
            print(f"Saved outputs to: {results_out_path}")

            tp_reg = 20
            heat_plot_stable = f"stability_top{tp_reg}_{group}_{met}"
            plot_least_stable_regions_heatmap(shared_map_file, movies, results_out_path, tp_reg, heat_plot_stable, group, met)
            
            heat_plot_summary = f"movie_specific_summary_{group}_{met}"
            plot_summary_heatmap(summary_file, results_out_path, heat_plot_summary)

# Execute script
if __name__ == "__main__":
    main()
