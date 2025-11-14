from dotenv import load_dotenv
load_dotenv()
import subprocess
from typing import List, Optional
import os

class GitFailedException(Exception):
    """Custom exception for Git command failures."""

class GitFailedToPushException(GitFailedException):
    """Custom exception for Git push failures."""    

def run_git_command(args: List[str], cwd: Optional[str] = None, raise_error: bool = True) -> subprocess.CompletedProcess:
    """Run a git command and return its output."""
    
    result = subprocess.run(['git'] + args, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if raise_error:
            raise GitFailedException(f"Git command failed: {' '.join(args)}\n{result.stderr}")
    return result

def pull_latest_changes(repo_path: str, branch: str = None) -> None:
    """Pull the latest changes from the remote repository."""
    args = ['pull']
    if branch:
        args.append('origin')
        args.append(branch)
    completed = run_git_command(args, cwd=repo_path, raise_error=False)
    if completed.returncode != 0:
        raise GitFailedException(f"Failed to pull latest changes:\n{completed.stderr}")
    return completed.stdout.strip()

def git_get_current_branch(repo_path: str) -> str:
    """Get the current git branch name."""
    completed = run_git_command(['branch', '--show-current'], cwd=repo_path)
    return completed.stdout.strip()


if __name__ == "__main__":
    _repo_path = '.'  # Current directory

    _repo_path = os.environ['DBASE_DIR']
    try:
        current_branch = git_get_current_branch(_repo_path)
        print(f"Current branch: {current_branch}")
        output = pull_latest_changes(_repo_path, branch=current_branch)
        print("Pull output:", output)
    except GitFailedException as e:
        print("Git operation failed:", e)