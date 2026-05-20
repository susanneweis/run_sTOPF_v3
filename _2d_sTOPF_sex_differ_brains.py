import os
import sys
import pandas as pd
import numpy as np

# path to your utils file/folder
sys.path.append(
    "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/code"
)

from _util_glass_brains_borders import create_glassbrains


def add_ratio_column(df):
    mean_within_sex_corr = (
        df["mean_female_female_corr"]
        + df["mean_male_male_corr"]
    ) / 2

    df["fm_separation_ratio"] = (
        1 - df["mean_female_male_corr"]
    ) / (
        1 - mean_within_sex_corr
    )

    df["fm_separation_ratio"] = df["fm_separation_ratio"].replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


def make_glassbrains_from_file(value_file,output_dir,prefix,roi_names,atlas_path,bot,top):
    glassbrain_cols = [
        "mean_female_female_corr",
        "mean_male_male_corr",
        "mean_female_male_corr",
        "fm_separation_ratio",
    ]

    for col, curr_bot, curr_top in zip(glassbrain_cols, bot, top):
        create_glassbrains(
            value_file,
            col,
            "region",
            roi_names,
            atlas_path,
            f"{prefix}: {col}",
            output_dir,
            f"{prefix}_{col}",
            "continuous",
            curr_bot,
            curr_top,
        )


def main(base_path,proj,code,nn,roi_names,atlas_path): 

    input_path = f"{base_path}/results_run_sTOPF_{code}_data_{proj}/results_nn{nn}/within_and_between_similarity_nn{nn}"
    input_file = f"{input_path}/ALL_movies_mean_pairwise_corr_by_sex_equalN.csv"

    output_dir = f"{input_path}/sex_differentiability_glassbrains"
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df = pd.read_csv(input_file)

    # --------------------------------------------------
    # Exclude concat and rest movies
    # --------------------------------------------------
    exclude_movies = ["concat", "REST1", "REST2", "rest1", "rest2"]
    df = df[~df["movie"].isin(exclude_movies)].copy()

    # --------------------------------------------------
    # Across-movie mean per region
    # --------------------------------------------------
    mean_df = (
        df
        .groupby("region", as_index=False)[
            [
                "mean_female_female_corr",
                "mean_male_male_corr",
                "mean_female_male_corr",
            ]
        ]
        .mean()
    )

    mean_df = add_ratio_column(mean_df)

    mean_file = os.path.join(
        output_dir,
        "mean_across_movies_pairwise_corr_by_sex_regions.csv"
    )
    mean_df.to_csv(mean_file, index=False)
    print(f"Saved: {mean_file}")
    
    prefix ="mean_across_movies"
    value_file = mean_file
    bot = [0, 0, 0, 0.98]
    top = [0.4, 0.4, 0.4, 1.02]

    make_glassbrains_from_file(value_file,output_dir,prefix,roi_names,atlas_path,bot,top)

    # --------------------------------------------------
    # Movie-wise glass brains
    # --------------------------------------------------
    moviewise_dir = os.path.join(output_dir, "moviewise")
    os.makedirs(moviewise_dir, exist_ok=True)

    for movie, movie_df in df.groupby("movie"):

        movie_df = movie_df.copy()
        movie_df = add_ratio_column(movie_df)

        movie_out_dir = os.path.join(moviewise_dir, movie)
        os.makedirs(movie_out_dir, exist_ok=True)

        movie_file = os.path.join(
            movie_out_dir,
            f"{movie}_pairwise_corr_by_sex_regions.csv"
        )
        movie_df.to_csv(movie_file, index=False)
        print(f"Saved: {movie_file}")

        bot = [0, 0, 0, 0.98]
        top = [0.4, 0.4, 0.4, 1.02]

        make_glassbrains_from_file(movie_file,movie_out_dir, movie,roi_names, atlas_path, bot, top)


if __name__ == "__main__":
    main()