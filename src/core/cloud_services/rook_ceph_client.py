import boto3
from pathlib import Path

class RookCephClient:
    def __init__(self):
        self.s3 = boto3.client("s3")

    def upload_to_s3(
        self, 
        bucket_name: str, 
        file_path: Path
    ) -> None:
        self.s3.upload_file(Bucket=bucket_name, Filename=str(file_path), Key=file_path.name)
        print(f"Uploaded {file_path} to s3://{bucket_name}/{file_path.name}")   

    def load_from_s3(
        self, 
        bucket_name: str, 
        file_path: Path
    ) -> None:
        self.s3.download_file(Bucket=bucket_name, Filename=file_path, Key=file_path.name,)
        print(f"Downloaded {file_path} from s3://{bucket_name}/{file_path.name}")

    def list_files(
        self, 
        bucket_name: str
    ) -> list[str]:
        response = self.s3.list_objects_v2(Bucket=bucket_name)
        return [obj["Key"] for obj in response.get("Contents", [])]

