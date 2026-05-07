from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import entropy
from tqdm import tqdm

PATH_LIST = [
    Path("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-12-02_012300/prediction_query_based_ensemble"), #ensemble prediction saved path
    Path("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-12-02_012515/prediction_query_based_ensemble"),
    Path("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-12-02_012601/prediction_query_based_ensemble"),
]

SAVE_PATH = Path("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/Final_query_based_ensemble")
print("Output location exist")

scans = sorted(list(PATH_LIST[0].glob("*/*.npy")))
# scans = sorted(list(PATH_LIST[0].glob("*/*.txt")))
print("scaning possible")


def process_scan(path, PATH_LIST, SAVE_PATH):
    """Processes a single scan file and saves the entropy values."""
    sequence = path.parent.stem
    file = path.name

    confid_list = []
    print("Starting to process scan...")
    for model in PATH_LIST:
        filepath = model / sequence / file
        confid_list.append(np.load(filepath))
        #confid_list.append(np.loadtxt(filepath))

    all_confids = np.stack(confid_list)
    avg_probs = np.mean(all_confids, 0)
    entropy_values = entropy(avg_probs, axis=1) / np.log(avg_probs.shape[1])

    if not (SAVE_PATH / sequence).exists():
        (SAVE_PATH / sequence).mkdir(parents=True, exist_ok=True)
    np.save(SAVE_PATH / sequence / file, entropy_values)
    return None  # Return None to avoid collecting results


if __name__ == "__main__":
    Parallel(n_jobs=8)(
        delayed(process_scan)(path, PATH_LIST, SAVE_PATH) for path in tqdm(scans)
    )

#=======================================================================================