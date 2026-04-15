import os
import re
import pandas as pd
import matplotlib.pyplot as plt



# =========================
# Helpers
# =========================
def safe_filename(s):
    """Make strings safe for filenames."""
    s = str(s)
    s = re.sub(r"[^\w\-_\. ]", "_", s)
    s = s.replace(" ", "_")
    return s


def get_colors(sex_series):
    """
    Map sex to colors.
    female -> pink/red
    male   -> blue
    """
    color_map = {
        "female": "#e75480",   # pink
        "male": "#1f77b4"      # blue
    }
    return sex_series.str.lower().map(color_map).fillna("gray")


def make_scatter_plot(
    data,
    x_col,
    y_col,
    title,
    out_file,
    xlabel=None,
    ylabel=None,
    alpha=0.7,
    point_size=20,
    show_legend=True
):
    """Create and save one scatter plot."""
    plt.figure(figsize=(6, 6))

    colors = get_colors(data["sex"])

    plt.scatter(
        data[x_col],
        data[y_col],
        c=colors,
        alpha=alpha,
        s=point_size,
        edgecolors="none"
    )

    # compute N and Pearson r
    valid = data[[x_col, y_col]].dropna()
    n = len(valid)

    if n > 1 and valid[x_col].std() > 0 and valid[y_col].std() > 0:
        r = valid[x_col].corr(valid[y_col])
        r_text = f"r = {r:.2f}"
    else:
        r_text = "r = NA"

    # add text box in plot
    plt.text(
        0.03, 0.97,
        f"n = {n}\n{r_text}",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.8,
            edgecolor="none"
        )
    )

    # diagonal reference line
    valid_xy = data[[x_col, y_col]].dropna()
    if not valid_xy.empty:
        x_min = min(valid_xy[x_col].min(), valid_xy[y_col].min())
        x_max = max(valid_xy[x_col].max(), valid_xy[y_col].max())
        plt.plot([x_min, x_max], [x_min, x_max], linestyle="--", linewidth=1)

    plt.xlabel(xlabel if xlabel is not None else x_col)
    plt.ylabel(ylabel if ylabel is not None else y_col)
    plt.title(title)

    if show_legend:
        # custom legend
        plt.scatter([], [], c="#e75480", label="female", s=40)
        plt.scatter([], [], c="#1f77b4", label="male", s=40)
        plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def make_metric_plots(
    base_outdir,
    df,
    x_col,
    y_col,
    metric_folder_name,
    x_label,
    y_label,
):
    """
    For one metric pair:
    - save one combined plot per movie across all regions
    - plus female-only and male-only versions
    - save one combined plot per region within each movie subfolder
    - plus female-only and male-only versions
    """
    metric_outdir = os.path.join(base_outdir, metric_folder_name)
    os.makedirs(metric_outdir, exist_ok=True)

    movies = sorted(df["movie"].dropna().unique())

    for movie in movies:
        movie_df = df[df["movie"] == movie].copy()
        if movie_df.empty:
            continue

        movie_df_female = movie_df[movie_df["sex"] == "female"].copy()
        movie_df_male = movie_df[movie_df["sex"] == "male"].copy()

        # -------------------------
        # 1) overview plot per movie
        # saved directly in metric_outdir
        # -------------------------
        movie_plot_file = os.path.join(
            metric_outdir,
            f"{safe_filename(movie)}_{metric_folder_name}_all_regions.png"
        )

        make_scatter_plot(
            data=movie_df,
            x_col=x_col,
            y_col=y_col,
            title=f"{movie}: {x_label} vs {y_label}\n(all regions, all participants)",
            out_file=movie_plot_file,
            xlabel=x_label,
            ylabel=y_label,
            alpha=0.2,
            point_size=18,
            show_legend=True
        )

        # female-only overview
        if not movie_df_female.empty:
            movie_plot_file_female = os.path.join(
                metric_outdir,
                f"{safe_filename(movie)}_{metric_folder_name}_all_regions_female_only.png"
            )

            make_scatter_plot(
                data=movie_df_female,
                x_col=x_col,
                y_col=y_col,
                title=f"{movie}: {x_label} vs {y_label}\n(all regions, female only)",
                out_file=movie_plot_file_female,
                xlabel=x_label,
                ylabel=y_label,
                alpha=0.2,
                point_size=18,
                show_legend=False
            )

        # male-only overview
        if not movie_df_male.empty:
            movie_plot_file_male = os.path.join(
                metric_outdir,
                f"{safe_filename(movie)}_{metric_folder_name}_all_regions_male_only.png"
            )

            make_scatter_plot(
                data=movie_df_male,
                x_col=x_col,
                y_col=y_col,
                title=f"{movie}: {x_label} vs {y_label}\n(all regions, male only)",
                out_file=movie_plot_file_male,
                xlabel=x_label,
                ylabel=y_label,
                alpha=0.2,
                point_size=18,
                show_legend=False
            )

        # -------------------------
        # 2) region-specific plots
        # saved in movie subfolder
        # -------------------------
        movie_region_dir = os.path.join(metric_outdir, safe_filename(movie))
        os.makedirs(movie_region_dir, exist_ok=True)

        regions = sorted(movie_df["region"].dropna().unique())

        for region in regions:
            region_df = movie_df[movie_df["region"] == region].copy()

            if region_df.empty:
                continue

            region_df_female = region_df[region_df["sex"] == "female"].copy()
            region_df_male = region_df[region_df["sex"] == "male"].copy()

            # combined region plot
            region_plot_file = os.path.join(
                movie_region_dir,
                f"{safe_filename(movie)}_region_{safe_filename(region)}_{metric_folder_name}.png"
            )

            make_scatter_plot(
                data=region_df,
                x_col=x_col,
                y_col=y_col,
                title=f"{movie} | Region {region}\n{x_label} vs {y_label}",
                out_file=region_plot_file,
                xlabel=x_label,
                ylabel=y_label,
                alpha=0.8,
                point_size=28,
                show_legend=True
            )

            # female-only region plot
            if not region_df_female.empty:
                region_plot_file_female = os.path.join(
                    movie_region_dir,
                    f"{safe_filename(movie)}_region_{safe_filename(region)}_{metric_folder_name}_female_only.png"
                )

                make_scatter_plot(
                    data=region_df_female,
                    x_col=x_col,
                    y_col=y_col,
                    title=f"{movie} | Region {region}\n{x_label} vs {y_label} (female only)",
                    out_file=region_plot_file_female,
                    xlabel=x_label,
                    ylabel=y_label,
                    alpha=0.8,
                    point_size=28,
                    show_legend=False
                )

            # male-only region plot
            if not region_df_male.empty:
                region_plot_file_male = os.path.join(
                    movie_region_dir,
                    f"{safe_filename(movie)}_region_{safe_filename(region)}_{metric_folder_name}_male_only.png"
                )

                make_scatter_plot(
                    data=region_df_male,
                    x_col=x_col,
                    y_col=y_col,
                    title=f"{movie} | Region {region}\n{x_label} vs {y_label} (male only)",
                    out_file=region_plot_file_male,
                    xlabel=x_label,
                    ylabel=y_label,
                    alpha=0.8,
                    point_size=28,
                    show_legend=False
                )


def main(base_path, proj, code, nn_mi):
# =========================
# Main
# =========================

    results_path = f"{base_path}/results_run_sTOPF_{code}_data_{proj}/results_nn{nn_mi}"
    base_outdir = f"{results_path}/ind_expr_scatterplots_nn{nn_mi}"

    input_file = f"{results_path}/individual_expression_all_nn{nn_mi}.csv"

    df = pd.read_csv(input_file)

    # Basic checks
    required_cols = [
        "subject",
        "sex",
        "movie",
        "region",
        "correlation_female",
        "correlation_male",
        "fem_mi",
        "mal_mi",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only rows with needed values
    df_corr = df.dropna(subset=["sex", "movie", "region", "correlation_female", "correlation_male"]).copy()
    df_mi   = df.dropna(subset=["sex", "movie", "region", "fem_mi", "mal_mi"]).copy()

    # Normalize sex labels just in case
    df_corr["sex"] = df_corr["sex"].astype(str).str.lower()
    df_mi["sex"]   = df_mi["sex"].astype(str).str.lower()

    # Correlation plots
    make_metric_plots(
        base_outdir,
        df=df_corr,
        x_col="correlation_female",
        y_col="correlation_male",
        metric_folder_name="corr_female_vs_male",
        x_label="Correlation to female typical response",
        y_label="Correlation to male typical response",
    )

    # MI plots
    make_metric_plots(
        base_outdir,
        df=df_mi,
        x_col="fem_mi",
        y_col="mal_mi",
        metric_folder_name="mi_female_vs_male",
        x_label="Mutual information to female typical response",
        y_label="Mutual information to male typical response"
    )

    print(f"Done. Plots saved to:\n{base_outdir}")


# Execute script
if __name__ == "__main__":
    main()