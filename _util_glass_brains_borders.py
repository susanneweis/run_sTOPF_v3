import pandas as pd
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from nilearn.plotting import plot_glass_brain
from nilearn.plotting import plot_stat_map
from matplotlib import cm
from matplotlib import colors

# Assign unique numerical IDs to each region name and adds them as a new column
def assign_roi_ids(df, roi_names, val_rois):
    """
    df: DataFrame containing a column 'roi_name'
    roi_names: list of ROI names in canonical order
               (index 0 -> ROI_ID 1)
    """
    # build lookup: roi_name -> ROI_ID (1-based)
    region_to_id = {region: i + 1 for i, region in enumerate(roi_names)}

    # map ROI_IDs
    df = df.copy()
    df['ROI_ID'] = df[val_rois].map(region_to_id)

    return df, region_to_id

def create_img_for_glassbrain_plot(stat_to_plot, atlas_path, n_roi):
    # Load atlas
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    
    # Create empty output image
    new_img = np.zeros(atlas_data.shape)
    
    # Sanity checks
    a_roi = int(np.max(atlas_data))
    if n_roi != a_roi:
        print(f"Mismatch between input ROIs ({n_roi}) and atlas ROIs ({a_roi})")

    # Check if stat_to_plot has the expected length
    if len(stat_to_plot) != n_roi:
        raise ValueError(f"Length of stat_to_plot ({len(stat_to_plot)}) does not match expected n_roi ({n_roi})")

    # Reshape data if needed
    stat_to_plot = np.reshape(stat_to_plot, (n_roi, 1))

    # Assign values to ROIs
    for roi in range(n_roi):
        voxel_indices = np.where(atlas_data == roi + 1)  # 1-based indexing
        if voxel_indices[0].size == 0:
            print(f"ROI {roi+1} not found in atlas.")
        new_img[voxel_indices] = stat_to_plot[roi]

    # Return Nifti image
    img_nii = nib.Nifti1Image(new_img, atlas_img.affine)
    return img_nii

def fill_glassbrain(n_r,res_df,column):
    # Initialize array for all ROIs
    roi_values = np.full(n_r, np.nan)

    # Fill in corr (convert Region to 0-based index)
    for _, row in res_df.iterrows():
        region_index = int(row['ROI_ID']) - 1  
        if 0 <= region_index < n_r:
            roi_values[region_index] = row[column]
    return roi_values

def create_glassbrains(value_file, value_name, value_roi_name, roi_names, at_path, title_str,out_path, name, cmap_mode, bot, top):

    roi_data = pd.read_csv(value_file)
        
    n_roi = roi_data[value_roi_name].nunique()
        
    cluster_brain, region_to_id_f  = assign_roi_ids(roi_data, roi_names, value_roi_name)
        
    roi_values = fill_glassbrain(n_roi,cluster_brain,value_name)

    if roi_values.max() > roi_values.min():
            
        output_file = f"{out_path}/glassbrain_{name}.png"
        output_file_sl = f"{out_path}/slices_{name}.png"

        # Create image
        img = create_img_for_glassbrain_plot(roi_values, at_path, n_roi)

        if cmap_mode == "discrete":

            # roi_values should be integer-ish cluster IDs
            n_labels = int(len(np.unique(roi_values[np.isfinite(roi_values)])))

            cmap = colors.ListedColormap(
                np.vstack([
                    plt.cm.tab20(np.linspace(0, 1, 20)),
                    plt.cm.tab20b(np.linspace(0, 1, 20)),
                ])[:n_labels]
            )

        elif cmap_mode == "continuous":

            # Good continuous choice for symmetric data
            cmap = plt.cm.RdBu_r

        else:
            raise ValueError("cmap_mode must be 'discrete' or 'continuous'")

        plot_glass_brain(
            img,
            cmap=cmap,
            #vmin=roi_values.min(),
            #vmax=roi_values.max(),
            vmin=bot,
            vmax=top,
            colorbar=True,
            title=title_str,
            plot_abs=False
        )  

        plt.savefig(output_file, bbox_inches='tight',dpi=300)
        plt.close()

        plot_stat_map(
            img,
            vmax=roi_values.max(),
            cmap=cmap,
            display_mode="mosaic",
            colorbar=True,
            title=title_str
        )

        plt.savefig(output_file_sl, bbox_inches='tight',dpi=300)
        plt.close()
    
        print(f"Saved brain map: {output_file}")

