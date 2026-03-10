from pathlib import Path
from tqdm import tqdm

from tsdae import TSDAE
import nltk

from test_2_CVE_db import iter_cve_json, extract_text

# --- NLTK setup (reuse your custom dir if you want) ---
nltk_data_dir = Path(__file__).parent.parent / ".venv" / "nltk_data"
nltk_data_dir.mkdir(parents=True, exist_ok=True)

nltk.data.path.insert(0, str(nltk_data_dir))
try:
    nltk.data.find("tokenizers/punkt_tab/english")
except LookupError:
    nltk.download("punkt_tab", download_dir=str(nltk_data_dir))

# --- Load CVE corpus ---
dataset_root = (
    Path(__file__).parent.parent.parent.parent / "datasets/cvelistV5-main"
)
print(f"Loading dataset at {dataset_root}")

ids, corpus = [], []
print("Start JSON iteration")
for _, j in tqdm(iter_cve_json(dataset_root, 2013, 2018)):
    cve_id, title, text = extract_text(j)
    ids.append(cve_id)
    corpus.append(title + "\n" + text)

if not corpus:
    raise SystemExit("No CVE JSON files found.")

# --- TSDAE training ---
model_name = "sentence-transformers/all-MiniLM-L6-v2"

tsdae = TSDAE(
    model_name=model_name,
    # you can tweak these hyperparameters:
    max_seq_length=256,
    corruption_rate=0.3,
)

# tsdae expects a list of sentences; corpus is already a list[str]
train_dataset = tsdae.load_dataset_from_list(corpus)

output_path = Path(__file__).parent.parent / "domain_adapted_model_tsdae"
output_path.mkdir(exist_ok=True)

model = tsdae.train(
    train_dataset=train_dataset,
    output_path=str(output_path),
    num_epochs=1,
    batch_size=8,
    learning_rate=3e-5,
)

print(f"TSDAE model saved to: {output_path}")
