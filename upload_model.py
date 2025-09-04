from huggingface_hub import HfApi
api = HfApi()

local_model_path = r"C:\Users\user\OneDrive\Desktop\DL- Projects\Dog-vs-Cat\Backend\dog_cat_model.h5"

repo_id = "Subhankar002/dog-vs-cat-classifier"

api.upload_file(
    path_or_fileobj=local_model_path,
    path_in_repo="dog_cat_model.h5",  
    repo_id=repo_id,
    repo_type="model"
)

print("Model uploaded successfully to Hugging Face:", repo_id)
