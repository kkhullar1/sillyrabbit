from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "kkhullar/deberta-pdtb-original-checkpoint-1590"

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = (
    PROJECT_DIR
    / "discourse"
    / "models"
    / "original_deberta_pdtb"
    / "checkpoint-1590"
)

MODEL_FILES = [
    "config.json",
    "pytorch_model.bin",
]


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for filename in MODEL_FILES:
        print(f"Downloading {filename}...")

        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=MODEL_DIR,
        )

    print("\nPDTB model setup complete.")
    print(f"Model directory: {MODEL_DIR}")


if __name__ == "__main__":
    main()