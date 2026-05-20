import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

# Input folder containing movie TSV files
data_dir = "/Users/sweis/Data/Arbeit/Juseless/data/project/brainvar_sexdiff_movies/data_run_sTOPF_v4/fMRIdata"

# Output folder
out_dir = os.path.join(data_dir, "SNR_results")
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# SCHAEFER 17 NETWORK LABELS
# ============================================================

network_names = [
    "VisCent",
    "VisPeri",
    "SomMotA",
    "SomMotB",
    "DorsAttnA",
    "DorsAttnB",
    "SalVentAttnA",
    "SalVentAttnB",
    "LimbicA",
    "LimbicB",
    "ContA",
    "ContB",
    "ContC",
    "DefaultA",
    "DefaultB",
    "DefaultC",
    "TempPar",
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_network(region_name):
    """
    Extract network name from region name.
    Regions without a Schaefer network become 'subcortical'.
    """
    for net in network_names:
        if net in region_name:
            return net
    return "subcortical"

def main(base_path, proj, code): 
    # ============================================================
    # SETTINGS
    # ============================================================

    data_path = f"{base_path}/data_run_sTOPF_{proj}"
    data_dir = f"{data_path}/fMRIdata"

    out_dir = f"{base_path}/results_run_sTOPF_{code}_data_{proj}/SNR_results"
    os.makedirs(out_dir, exist_ok=True)
    # ============================================================
    # FIND FILES
    # ============================================================

    tsv_files = sorted(glob.glob(os.path.join(data_dir, "*.tsv")))

    print(f"Found {len(tsv_files)} movie files")

    all_region_snr = []
    all_network_snr = []

    # ============================================================
    # PROCESS EACH MOVIE
    # ============================================================

    for file in tsv_files:

        movie_name = os.path.basename(file)
        movie_name = movie_name.replace("BOLD_Schaefer_436_2025_mean_aggregation_task-", "")
        movie_name = movie_name.replace("_MOVIES.tsv", "")

        print(f"\nProcessing movie: {movie_name}")

        # --------------------------------------------------------
        # LOAD DATA
        # --------------------------------------------------------

        df = pd.read_csv(file, sep="\t")

        subject_col = "subject"

        # --------------------------------------------------------
        # IDENTIFY ROI COLUMNS
        # --------------------------------------------------------

        non_roi_cols = [
            subject_col,
            "timepoint",
        ]

        roi_cols = [c for c in df.columns if c not in non_roi_cols]

        print(f"Number of ROI columns: {len(roi_cols)}")

        # --------------------------------------------------------
        # COMPUTE SNR
        # mean(timecourse) / std(timecourse)
        # per subject x region
        # --------------------------------------------------------

        region_results = []

        for subject, sub_df in df.groupby(subject_col):

            for roi in roi_cols:

                tc = pd.to_numeric(sub_df[roi], errors="coerce").dropna()

                if len(tc) == 0:
                    continue

                mean_signal = tc.mean()
                noise = tc.std()

                if noise == 0:
                    snr = np.nan
                else:
                    snr = mean_signal / noise

                network = get_network(roi)

                region_results.append({
                    "movie": movie_name,
                    "subject": subject,
                    "Region": roi,
                    "Network": network,
                    "mean_signal": mean_signal,
                    "noise_sd": noise,
                    "snr": snr,
                })

        region_df = pd.DataFrame(region_results)

        # --------------------------------------------------------
        # SAVE REGION-LEVEL SNR
        # --------------------------------------------------------

        region_out = os.path.join(
            out_dir,
            f"{movie_name}_region_SNR.csv"
        )

        region_df.to_csv(region_out, index=False)

        all_region_snr.append(region_df)

        # --------------------------------------------------------
        # NETWORK-LEVEL SUMMARY
        # average across regions within network
        # --------------------------------------------------------

        network_df = (
            region_df
            .groupby(["movie", "subject", "Network"])["snr"]
            .mean()
            .reset_index()
        )

        network_out = os.path.join(
            out_dir,
            f"{movie_name}_network_SNR.csv"
        )

        network_df.to_csv(network_out, index=False)

        all_network_snr.append(network_df)

        # --------------------------------------------------------
        # PLOT
        # Separate plot for each movie
        # --------------------------------------------------------

        plot_df = network_df.copy()

        network_order = (
            plot_df
            .groupby("Network")["snr"]
            .mean()
            .sort_values(ascending=False)
            .index
        )

        plt.figure(figsize=(14, 7))

        # --------------------------------------------------------
        # Scatter all subjects
        # --------------------------------------------------------

        for i, net in enumerate(network_order):

            vals = plot_df.loc[
                plot_df["Network"] == net,
                "snr"
            ].values

            x = np.random.normal(i, 0.06, size=len(vals))

            plt.scatter(
                x,
                vals,
                alpha=0.5,
                s=25,
            )

        # --------------------------------------------------------
        # Mean + SD
        # --------------------------------------------------------

        summary = (
            plot_df
            .groupby("Network")["snr"]
            .agg(["mean", "std"])
            .reindex(network_order)
        )

        plt.errorbar(
            x=np.arange(len(summary)),
            y=summary["mean"],
            yerr=summary["std"],
            fmt="o",
            capsize=5,
            linewidth=2,
        )

        plt.xticks(
            np.arange(len(summary)),
            summary.index,
            rotation=45,
            ha="right"
        )

        plt.ylabel("Signal-to-noise ratio (mean / SD)")
        plt.title(f"SNR across networks — {movie_name}")

        plt.tight_layout()

        plot_file = os.path.join(
            out_dir,
            f"{movie_name}_network_SNR_plot.png"
        )

        plt.savefig(plot_file, dpi=300)
        plt.close()

        print(f"Saved plot: {plot_file}")

    # ============================================================
    # COMBINED OUTPUTS
    # ============================================================

    all_region_df = pd.concat(all_region_snr, ignore_index=True)
    all_network_df = pd.concat(all_network_snr, ignore_index=True)

    all_region_df.to_csv(
        os.path.join(out_dir, "ALL_movies_region_SNR.csv"),
        index=False
    )

    all_network_df.to_csv(
        os.path.join(out_dir, "ALL_movies_network_SNR.csv"),
        index=False
    )

    print("\nDone.")
    print(f"Results saved to:\n{out_dir}")