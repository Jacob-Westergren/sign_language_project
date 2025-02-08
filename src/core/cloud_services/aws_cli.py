import subprocess   # library for running shell commands in python code
from pathlib import Path

"""
AWS S3 utility functions that use the AWS CLI, for info check out:
https://docs.aws.amazon.com/cli/latest/userguide/cli-services-s3-commands.html
"""

def upload_file_to_s3(
    bucket_name: str, 
    file_path: Path
) -> None:
    cmd = f"aws s3 cp {file_path} s3://{bucket_name}/{file_path.name}"
    
    subprocess.run(cmd, shell=True)
    print(f"Uploaded {file_path} to s3://{bucket_name}/{file_path.name}")

def upload_directory_to_s3(
    bucket_name: str, 
    directory_path: Path, 
    exclude_files: str = ""
) -> None:

    cmd = f"aws s3 cp {directory_path} s3://{bucket_name}/{directory_path.name} --recursive"
    if exclude_files: cmd += f" --exclude {exclude_files}"    # ex. "*.mp4" to only upload frames
    
    subprocess.run(cmd, shell=True)
    print(f"Uploaded {directory_path} to s3://{bucket_name}/{directory_path.name}")

def download_file_from_s3(
    bucket_name: str, 
    file_path: Path
) -> None:
    cmd = f"aws s3 cp s3://{bucket_name}/{file_path.name} {file_path}"
    
    subprocess.run(cmd, shell=True)
    print(f"Downloaded {file_path} from s3://{bucket_name}/{file_path.name}")

def download_directory_from_s3(
    bucket_name: str, 
    output_dir: Path, 
    exclude_files: str = ""
) -> None:

    cmd = f"aws s3 cp s3://{bucket_name} {output_dir} --recursive"
    if exclude_files: cmd += f" --exclude {exclude_files}"    # ex. "*.mp4" to only download frames
    
    subprocess.run(cmd, shell=True)
    print(f"Downloaded all files from s3://{bucket_name} to {output_dir}")

def list_files(
    bucket_name: str
) -> list[str]:
    # run the command and store it as a string
    cmd = f"aws s3 ls s3://{bucket_name}"
    output = subprocess.check_output(cmd, shell=True).decode("utf-8") 

    # split the output into a list of strings, where each string is a line of the output
    return output.split("\n")   

def list_buckets() -> list[str]:
    # run the command and store it as a string
    cmd = "aws s3 ls"
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    return output.split("\n")

