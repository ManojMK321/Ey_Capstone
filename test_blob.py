"""
Quick test script - run from the project root:
    python test_blob.py
    python test_blob.py path/to/your_contract.pdf
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Always load .env from the same folder as this script
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ── sanity-check env vars before doing anything ──────────────────────────────
required = [
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_CONTAINER_NAME",
]
missing = [k for k in required if not os.getenv(k)]
if missing:
    print(f"[ERROR] Missing env vars: {', '.join(missing)}")
    print("       Fill them in your .env file and try again.")
    sys.exit(1)

from src.retrieval.blob_storage import (
    list_blobs,
    upload_blob,
    download_blob,
    blob_exists,
    delete_blob,
)


def test_connection():
    print("\n[1] Testing connection — listing blobs...")
    blobs = list_blobs()
    if blobs:
        print(f"    Found {len(blobs)} existing blob(s):")
        for b in blobs:
            print(f"      - {b}")
    else:
        print("    Container is empty (that's fine).")
    print("    Connection OK")


def test_upload(pdf_path: str):
    blob_name = os.path.basename(pdf_path)
    print(f"\n[2] Uploading '{blob_name}'...")
    with open(pdf_path, "rb") as f:
        data = f.read()
    url = upload_blob(blob_name, data)
    print(f"    Uploaded  -> {url}")

    print(f"\n[3] Checking blob exists...")
    assert blob_exists(blob_name), "blob_exists returned False after upload!"
    print("    Exists: True")

    print(f"\n[4] Downloading and verifying size...")
    downloaded = download_blob(blob_name)
    assert len(downloaded) == len(data), "Size mismatch after download!"
    print(f"    Downloaded {len(downloaded):,} bytes — matches original")

    print(f"\n[5] Listing blobs after upload...")
    blobs = list_blobs()
    print(f"    Blobs in container: {blobs}")

    return blob_name


def test_delete(blob_name: str):
    print(f"\n[6] Deleting test blob '{blob_name}'...")
    delete_blob(blob_name)
    assert not blob_exists(blob_name), "blob still exists after delete!"
    print("    Deleted successfully")


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None

    test_connection()

    if pdf_path:
        if not os.path.isfile(pdf_path):
            print(f"[ERROR] File not found: {pdf_path}")
            sys.exit(1)
        blob_name = test_upload(pdf_path)
        # comment out the next line if you want to keep the file in the container
        # test_delete(blob_name)
        print("\n All tests passed. The PDF is now in Azure Blob Storage.")
    else:
        print("\n Connection test passed.")
        print(" To also test upload, run:")
        print("     python test_blob.py path\\to\\contract.pdf")
