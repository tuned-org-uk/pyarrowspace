from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader
from pathlib import Path

from tqdm import tqdm
import nltk
nltk_data_dir = Path(__file__).parent.parent / ".venv"
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

from test_2_CVE_db import iter_cve_json, extract_text

dataset_root = Path(__file__).parent.parent.parent.parent / "datasets/cvelistV5-main"
print(f"Loading dataset at {dataset_root}")

ids, corpus = [], []
print("Start JSON iteration")
for _, j in tqdm(iter_cve_json(dataset_root, 2013, 2018)):
    cve_id, title, text = extract_text(j)
    ids.append(cve_id)
    corpus.append(title + "\n" + text)
if not corpus:
    raise SystemExit("No CVE JSON files found.")

model = SentenceTransformer("all-MiniLM-L6-v2")

train_dataset = DenoisingAutoEncoderDataset(corpus)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
train_loss = losses.DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    weight_decay=0,
    optimizer_params={"lr": 3e-5},
)

# --- Add the model saving step ---
output_path = Path("./domain_adapted_model")
output_path.mkdir(exist_ok=True) # Create the directory if it doesn't exist
model.save(output_path)

# --- Use the fine-tuned model to generate embeddings ---
domain_adapted_embeddings = model.encode(corpus)

print(f"Model saved to: {output_path}")

# # --- Example of loading the saved model ---
# from sentence_transformers import SentenceTransformer
# loaded_model = SentenceTransformer(output_path)
# print(f"Model successfully loaded from: {output_path}")
