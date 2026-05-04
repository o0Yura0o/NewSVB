from huggingface_hub import hf_hub_download
import os

local_dir = "c:/Users/neo29/workspace/SVC/NSVB-ZH/data/VocalVerse"
os.makedirs(f"{local_dir}/VocalVerse_Datasets-human_labels", exist_ok=True)

files = [
    "VocalVerse_Datasets-human_labels/Amateur_overall_mos_avg5.xlsx",
    "VocalVerse_Datasets-human_labels/Professional_multidim_annotations_raw_Timbre_Breath_Emotion_Technique.xlsx",
    "VocalVerse_Datasets-human_labels/Professional_scoring_rubric.xlsx",
]

for f in files:
    print(f"Downloading {f} ...")
    path = hf_hub_download(
        repo_id="karl-wang/VocalVerse-dataset",
        repo_type="dataset",
        filename=f,
        local_dir=local_dir,
    )
    print(f"  -> {path}")

print("Labels downloaded.")
