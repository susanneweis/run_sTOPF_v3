import os 
import pandas as pd

def main(base_path, proj):
    """
    Create a valid subject list with equal numbers of male and female subjects.

    This function reproduces the original subject-selection logic:
    1. keep only subjects present in Participant_sex_info.csv and complete_participants.csv
    2. remove subjects listed in excluded_participants.csv
    3. downsample the larger sex group so that N male == N female
    4. save the resulting list to CSV

    Gender coding is assumed to be: 1 = male, 2 = female.

    """
    data_path = f"{base_path}/data_run_sTOPF_{proj}"

    phenotype_path = f"{data_path}/Participant_sex_info.csv"
    complete_participants_path = f"{data_path}/complete_participants.csv"
    excluded_participants_path = f"{data_path}/excluded_participants.csv"

    output_filename="valid_subjects_balanced_sex.csv"
    output_path = f"{data_path}/{output_filename}"

   
    random_state=42

    for path in [phenotype_path, complete_participants_path, excluded_participants_path]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            raise FileNotFoundError

    phenotypes = pd.read_csv(phenotype_path)
    phenotypes.columns = ["subject_ID", "gender"]
    phenotypes["subject_ID"] = phenotypes["subject_ID"].astype(str)

    complete_participants = set(pd.read_csv(complete_participants_path)["subject"].astype(str))
    excluded_participants = set(pd.read_csv(excluded_participants_path)["subject"].astype(str))

    valid_subjects = complete_participants.intersection(set(phenotypes["subject_ID"]))
    valid_subjects = valid_subjects.difference(excluded_participants)

    valid_df = phenotypes[phenotypes["subject_ID"].isin(valid_subjects)].copy()

    sex_mapping = {1: 'male', 2: 'female'}

    valid_df['gender'] = valid_df['gender'].replace(sex_mapping)

    male_df = valid_df[valid_df["gender"] == 'male'].copy()
    female_df = valid_df[valid_df["gender"] == 'female'].copy()

    n_equal = min(len(male_df), len(female_df))

    male_df = male_df.sample(n=n_equal, random_state=random_state)
    female_df = female_df.sample(n=n_equal, random_state=random_state)

    balanced_df = (
        pd.concat([male_df, female_df], axis=0)
        .sort_values("subject_ID")
        .reset_index(drop=True)
    )

    balanced_df.to_csv(output_path, index=False)

    print(f"Original valid subjects after exclusion: {len(valid_df)}")
    print(f"Male subjects before balancing: {len(valid_df[valid_df['gender'] == 1])}")
    print(f"Female subjects before balancing: {len(valid_df[valid_df['gender'] == 2])}")
    print(f"Balanced valid subjects written: {len(balanced_df)}")
    print(f"Balanced N per sex: {n_equal}")
    print(f"Saved to: {output_path}")

    return output_path