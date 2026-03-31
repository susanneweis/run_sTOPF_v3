import os
import pandas as pd
from _util_glass_brains_borders import create_glassbrains

def main(base_path, proj, code, movies_properties):
    res_path = f"{base_path}/results_run_sTOPF_{code}_data_{proj}"
    sex_pca_path = f"{res_path}/results_PCA_per_sex"
    all_pca_path = f"{res_path}/results_PCA_all"
    glass_out_path = f"{res_path}/results_PC1_var_expl_brains"
    os.makedirs(glass_out_path, exist_ok=True)

    data_path = f"{base_path}/data_run_sTOPF_{proj}"
    atlas_path = f"{data_path}/Susanne_Schaefer_436.nii"
    roi_name_file = f"{data_path}/ROI_names.csv"
    roi_names = pd.read_csv(roi_name_file)["roi_name"].tolist()

    movies = list(movies_properties.keys())

    for curr_mov in movies: 
        
        for group in ["all", "female", "male"]:

            if group == "all":
                in_p = all_pca_path
            else:
                in_p = sex_pca_path
            
        
            cluster_assign_file = f"{in_p}/{curr_mov}/explained_variance_1_{group}_allROI.csv"
            title = f"PC1 Explained Variance {curr_mov} {group}"
            name_str = f"PC1_Exp_Var_{curr_mov}_{group}"
            create_glassbrains(cluster_assign_file, "explained_variance_1", "Region",roi_names, atlas_path, title, glass_out_path, name_str,"continuous",0,0.5)
    

if __name__ == "__main__":
    main()