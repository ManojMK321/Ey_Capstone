"""
Test the FastAPI upload endpoint.
Make sure the server is running first: python main.py

Usage:
    python test_api_upload.py
    python test_api_upload.py "C:\\path\\to\\file.pdf"
"""
import sys
import requests

BASE_URL = "http://localhost:8000"

def check_server():
    try:
        r = requests.get(f"{BASE_URL}/docs", timeout=3)
        print("[OK] Server is running")
        return True
    except Exception:
        print("[ERROR] Server is not running. Start it with: python main.py")
        return False

def list_blobs():
    r = requests.get(f"{BASE_URL}/upload/list")
    data = r.json()
    print(f"\n[Blob List] {data['count']} file(s) in storage:")
    for f in data["files"]:
        print(f"  - {f}")

def upload_pdf(pdf_path: str):
    print(f"\n[Upload] Uploading: {pdf_path}")
    with open(pdf_path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/upload/",
            files=[("files", (pdf_path.split("\\")[-1], f, "application/pdf"))],
        )
    if response.status_code == 200:
        data = response.json()
        print(f"[OK] {data['message']}")
        for file in data["files"]:
            print(f"  - {file['original_name']} -> {file['blob_url']}")
        if data["errors"]:
            print(f"[WARN] Errors: {data['errors']}")
    else:
        print(f"[ERROR] Status {response.status_code}: {response.text}")

if __name__ == "__main__":
    if not check_server():
        sys.exit(1)

    pdf = sys.argv[1] if len(sys.argv) > 1 else None

    if pdf:
        upload_pdf(pdf)
    else:
        print("No PDF path given. Usage: python test_api_upload.py path\\to\\file.pdf")

    list_blobs()
