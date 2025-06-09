import json
import sys

def strip_metadata(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    if "metadata" in nb:
        nb["metadata"].pop("kernelspec", None)
        nb["metadata"].pop("language_info", None)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Cleaned metadata in: {file_path}")

# 실행 예: python strip_kernel_metadata.py path/to/notebook.ipynb
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python strip_kernel_metadata.py <notebook_path>")
    else:
        strip_metadata(sys.argv[1])
