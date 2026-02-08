from pathlib import Path
from arrowspace import load_arrowspace

graph_params = {"eps": 0.97, "k": 21, "topk": 10, "p": 2.0, "sigma": 0.1}

print(f"Loading ArrowSpace from storage")
aspace, gl = load_arrowspace(
    storage_path=str(Path(__file__).parent.parent / "storage/"),
    dataset_name="dataset_858414",   # pick one dataset in storage/
    graph_params=graph_params,
    energy=False,
)