import os
from io import BytesIO
from pathlib import Path
from azure.storage.blob import BlobServiceClient, ContainerClient
from dotenv import load_dotenv

# Walk up from this file's location to find .env at the project root
load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env", override=True)


def _get_container_client(container: str = None) -> ContainerClient:
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = container or os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    client = BlobServiceClient.from_connection_string(conn_str)
    return client.get_container_client(container_name)


def list_blobs(container: str = None) -> list[str]:
    """Return a list of blob names in the container."""
    cc = _get_container_client(container)
    return [blob.name for blob in cc.list_blobs()]


def download_blob(blob_name: str, container: str = None) -> bytes:
    """Download a blob and return its raw bytes."""
    cc = _get_container_client(container)
    stream = BytesIO()
    cc.get_blob_client(blob_name).download_blob().readinto(stream)
    return stream.getvalue()


def upload_blob(blob_name: str, data: bytes, container: str = None, overwrite: bool = True) -> str:
    """Upload bytes to a blob and return its URL."""
    cc = _get_container_client(container)
    blob_client = cc.get_blob_client(blob_name)
    blob_client.upload_blob(data, overwrite=overwrite)
    return blob_client.url


def delete_blob(blob_name: str, container: str = None) -> None:
    """Delete a blob from the container."""
    cc = _get_container_client(container)
    cc.get_blob_client(blob_name).delete_blob()


def blob_exists(blob_name: str, container: str = None) -> bool:
    """Check whether a blob exists."""
    cc = _get_container_client(container)
    return cc.get_blob_client(blob_name).exists()
