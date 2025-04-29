import logging
from pathlib import Path
from git import Repo
import os
logger = logging.getLogger(__name__)

def clone_repo(
    repo_url: str,
    branch: str = "main",
    cloned_repo_base_dir: str = str(Path.cwd()),
) -> Path:
    """
    Clone or pull a repository.
    
    Args:
        repo_url: Git HTTPS/SSH URL
        branch: Branch to clone or update (default: main)
        cloned_repo_base_dir: Local directory to clone into (default: current working dir)
        
    Returns:
        Path to the cloned repository
    """
    repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    repo_path = os.path.join(cloned_repo_base_dir, repo_name)
    
    if not Path(repo_path).exists():
        logger.info("Cloning %s → %s", repo_url, repo_path)
        Repo.clone_from(repo_url, repo_path, depth=1, branch=branch)
    else:
        logger.info("Pulling latest on %s", repo_name)
        Repo(repo_path).remotes.origin.pull(branch)
        
    return Path(repo_path)


def push_to_repo(repo_path: Path) -> None:
    """
    Push the .kno folder back to the remote repository.
    
    Args:
        repo_path: Path to the cloned repository
    """
    repo = Repo(repo_path)
    kno_dir = os.path.join(repo_path, ".kno")
    try:
        logger.info("Pushing .kno to %s", repo_path)
        relative_kno = os.path.relpath(str(kno_dir), str(repo_path))
        repo.git.add(str(relative_kno))
        repo.index.commit("Add/update .kno embedding database")
        repo.remote().push()
    except Exception as e:
        logger.warning("Failed to push .kno to %s: %s", repo_path, e)
