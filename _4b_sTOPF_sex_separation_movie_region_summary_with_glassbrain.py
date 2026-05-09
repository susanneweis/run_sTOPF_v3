import os
import pandas as pd

from _util_glass_brains import create_glassbrains


def main(base_path, proj, code, roi_names, at_path):
    """
    Summarise sex separability across movies and regions, create one
    glass brain per movie for the sex separability score, and create one
    additional glass brain for the mean sex separability across movies.

    Parameters
    ----------
    base_path : str
        Base project path.
    proj : str
        Project/data version string used in results folder name.
    code : str
        sTOPF code/version string used in results folder name.
    roi_names : list[str]
        ROI names in the exact order of the atlas labels
        (atlas value 1 == roi_names[0], etc.).
    at_path : str
        Path to atlas NIfTI image with integer ROI labels.
    """
    results_path = f"{base_path}/results_run_sTOPF_{code}_data_{proj}"
    res_in_dir = f"{results_path}/sex_separability"

    out_dir = f"{res_in_dir}/movies_regions"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(f"{res_in_dir}/ALL_movies_sex_diff_scores.csv")
    df = df[~df["movie"].isin(["REST1", "REST2", "concat"])].copy()

    # Optional: exclude rest.
    # Keep/remove concat depending on whether you want one glass brain for concat too.
    #df = df[~df["movie"].isin(["REST1", "REST2"])].copy()

    # ------------------------------------------------------------------
    # Region-wise movie sensitivity
    # ------------------------------------------------------------------
    region_sens = (
        df.groupby("Region")["score"]
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

    region_sens.to_csv(
        os.path.join(out_dir, "region_movie_sensitivity_descriptive.csv"),
        index=False,
    )


    # ------------------------------------------------------------------
    # Mean sex separability across movies: one summary value per region
    # ------------------------------------------------------------------
    mean_across_movies = (
        df.groupby("Region", as_index=False)["score"]
        .mean()
        .rename(columns={"score": "mean_score_across_movies"})
        .sort_values("Region")
    )

    mean_across_movies_file = os.path.join(
        out_dir,
        "mean_sex_separability_across_movies_by_region.csv",
    )
    mean_across_movies.to_csv(mean_across_movies_file, index=False)

    create_glassbrains(
        value_file=mean_across_movies_file,
        value_name="mean_score_across_movies",
        value_roi_name="Region",
        roi_names=roi_names,
        at_path=at_path,
        title_str="Mean sex separability score across movies",
        out_path=out_dir,
        name="mean_sex_separability_score_across_movies",
        cmap_mode="continuous",
    )

    # ------------------------------------------------------------------
    # Movie-wise overall sex-difference strength
    # ------------------------------------------------------------------
    movie_summary = (
        df.groupby("movie")["score"]
        .agg(
            mean_score="mean",
            median_score="median",
            sd_across_regions="std",
            max_score="max",
            n_regions="count",
        )
        .reset_index()
        .sort_values("mean_score", ascending=False)
    )

    movie_summary.to_csv(
        os.path.join(out_dir, "movie_summary_scores.csv"),
        index=False,
    )

    # ------------------------------------------------------------------
    # Full clean matrix for heatmaps
    # ------------------------------------------------------------------
    matrix = df.pivot(index="Region", columns="movie", values="score")
    matrix.to_csv(os.path.join(out_dir, "region_by_movie_score_matrix.csv"))

    # ------------------------------------------------------------------
    # Glass brains: one sex-separability map per movie
    # ------------------------------------------------------------------
    per_movie_dir = os.path.join(out_dir, "per_movie_score_tables")
    os.makedirs(per_movie_dir, exist_ok=True)

    movies = sorted(df["movie"].dropna().unique())
    movies = [m for m in movies if m not in ["REST1", "REST2", "concat"]]

    for movie in movies:
        movie_df = df[df["movie"] == movie].copy()

        # Make sure there is exactly one row per region for this movie.
        movie_df = (
            movie_df.groupby("Region", as_index=False)["score"]
            .mean()
            .sort_values("Region")
        )

        movie_safe = str(movie).replace("/", "_").replace(" ", "_")
        movie_file = os.path.join(
            per_movie_dir,
            f"sex_separability_score_{movie_safe}.csv",
        )
        movie_df.to_csv(movie_file, index=False)

        create_glassbrains(
            value_file=movie_file,
            value_name="score",
            value_roi_name="Region",
            roi_names=roi_names,
            at_path=at_path,
            title_str=f"Sex separability score - {movie}",
            out_path=out_dir,
            name=f"sex_separability_score_{movie_safe}",
            cmap_mode="continuous",
        )

    print("Saved outputs to:", out_dir)
    print("Saved per-movie glass brains to:", os.path.join(out_dir, "glassbrains"))
    print("Saved per-movie and mean-across-movies glass brains to:", os.path.join(out_dir, "glassbrains"))
    print("Saved per-movie and mean-across-movies slice maps to:", os.path.join(out_dir, "slices"))
