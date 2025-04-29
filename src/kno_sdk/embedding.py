from typing import Dict, List, Tuple, Optional, Any, TypedDict, Union
from tree_sitter_languages import get_language
from tree_sitter import Parser
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from git import Repo
import logging
from pathlib import Path
import os
import time

from .constant import EXT_TO_LANG, LANG_NODE_TARGETS, TOKEN_LIMIT, MAX_FALLBACK_LINES, BINARY_EXTS, EmbeddingMethod
from .git import clone_repo, push_to_repo

Language = get_language
logger = logging.getLogger(__name__)


# ─────────── Load grammars (works w/ tree_sitter 0.20 → 0.22) ────────────
LANGUAGE_CACHE: Dict[str, Language] = {}
for lang_name in set(EXT_TO_LANG.values()):
    try:
        LANGUAGE_CACHE[lang_name] = Language(lang_name)
    except TypeError:
        logger.warning("No grammar for %s – falling back to line chunking", lang_name)


PARSER_CACHE: Dict[str, Parser] = {
    lang: (lambda l: (p := Parser(), p.set_language(l), p)[0])(lang_obj)
    for lang, lang_obj in LANGUAGE_CACHE.items()
}





class RepoIndex:
    path: Path
    vector_store: Chroma
    digest: dict[str, Any]

    def __init__(self, vector_store: Chroma, digest: str, path: Path = Path.cwd()):
        self.path = path
        self.vector_store = vector_store
        self.digest = digest


def _extract_semantic_chunks(path: Path, text: str) -> List[str]:
    lang_name = EXT_TO_LANG.get(path.suffix.lower())
    if not lang_name or lang_name not in PARSER_CACHE:
        return []
    parser = PARSER_CACHE[lang_name]
    tree = parser.parse(text.encode())
    targets = LANG_NODE_TARGETS.get(lang_name, ())
    chunks: List[str] = []

    def walk(node):
        if node.type in targets:
            code = text[node.start_byte : node.end_byte]
            header = f"// {path.name}:{node.start_point[0]+1}-{node.end_point[0]+1}\n"
            chunks.append(header + code)
            return
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return chunks


def _fallback_line_chunks(path: Path, text: str) -> List[str]:
    lines = text.splitlines()
    chunks = []
    for i in range(0, len(lines), MAX_FALLBACK_LINES):
        header = f"// {path}:{i+1}-{min(i+MAX_FALLBACK_LINES,len(lines))}\n"
        body = "\n".join(lines[i : i + MAX_FALLBACK_LINES])
        chunks.append(header + body)
    return chunks


class SBERTEmbeddings(Embeddings):
    def __init__(self, model_name: str = "microsoft/graphcodebert-base"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

class JINAAIEmbeddings(Embeddings):
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-code"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.max_seq_length = 768

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()


def _build_directory_digest(
    repo_path: Path,
    skip_dirs: set[str],
    skip_files: set[str],
    max_depth: int = 5,
    max_chars: int = 10000
) -> dict[str, Any]:
    file_list: List[str] = []
    truncated = False

    for root, dirs, files in os.walk(repo_path):
        rel_root = Path(root).relative_to(repo_path)
        depth = len(rel_root.parts)

        if depth > max_depth or any(part in skip_dirs for part in rel_root.parts):
            dirs.clear()
            continue

        files = [f for f in files if f not in skip_files]
        for file_name in files:
            full_path = str(rel_root / file_name) if rel_root != Path(".") else file_name
            file_list.append(full_path)

        if sum(len(f) + 4 for f in file_list) > max_chars:
            truncated = True
            file_list.append("...")
            break

    output = {
        "summary": {
            "total_files": len(file_list) - (1 if truncated else 0),
            "truncated": truncated
        },
        "files": file_list
    }

    # Pretty-print JSON inside Python string with newlines and indentation
    return output


# 3) parse out the timestamp and pick the max
def _ts(d: Path) -> int:
    parts = d.name.split("_")
    try:
        return int(parts[2])
    except (IndexError, ValueError):
        return 0

def index_repo(
    repo_path: Path,
    embedding: Union[EmbeddingMethod, str] = EmbeddingMethod.SBERT,
) -> RepoIndex:
    """
    Index a repository that has already been cloned.
    
    Args:
        repo_path: Path to the cloned repository
        embedding: EmbeddingMethod.OPENAI or EmbeddingMethod.SBERT, or their string values
        
    Returns:
        RepoIndex object with the vector store and repository information
    """
    if isinstance(embedding, str):
        try:
            embedding = EmbeddingMethod(embedding)  # Convert string to enum
        except ValueError:
            raise ValueError(f"Invalid embedding method: {embedding}. Must be one of {[e.value for e in EmbeddingMethod]}")

    repo = Repo(repo_path)
    repo_name = repo_path.name
    kno_dir = os.path.join(repo_path, ".kno")
    skip_dirs = {".git", "node_modules", "build", "dist", "target", ".vscode", ".kno"}
    skip_files = {"package-lock.json", "yarn.lock", ".prettierignore"}
    digest = _build_directory_digest(repo_path, skip_dirs, skip_files)

    # 2. choose embedding
        
    if embedding.value == "OpenAIEmbedding":
        embed_fn = OpenAIEmbeddings()
    elif embedding.value == "SBERTEmbedding":
        embed_fn = SBERTEmbeddings()
    else:
        embed_fn = JINAAIEmbeddings()
    
    
    commit = repo.head.commit.hexsha[:7]
    time_ms = int(time.time() * 1000)
    subdir = f"embedding_{embedding.value}_{time_ms}_{commit}"
    
    vs = Chroma(
        collection_name=repo_name,
        embedding_function=embed_fn,
        persist_directory=os.path.join(kno_dir, subdir),
    )

    # 3. index if empty
    if vs._collection.count() == 0:
        logger.info("Indexing %s …", repo_name)
        texts, metas = [], []

        for fp in Path(repo_path).rglob("*.*"):
            if any(p in skip_dirs for p in fp.parts) or fp.name in skip_files:
                continue
            if fp.stat().st_size > 2_000_000 or fp.suffix.lower() in BINARY_EXTS:
                continue
            content = fp.read_text(errors="ignore")
            chunks = _extract_semantic_chunks(fp, content) or _fallback_line_chunks(
                fp, content
            )
            for chunk in chunks:
                texts.append(chunk[:TOKEN_LIMIT])
                metas.append({"source": str(fp.relative_to(repo_path))})
        vs.add_texts(texts=texts, metadatas=metas)
        logger.info("Embedded %d chunks", len(texts))

    return RepoIndex(vector_store=vs, digest=digest, path=repo_path)


def clone_and_index(
    repo_url: str,
    branch: str = "main",
    embedding: Union[EmbeddingMethod, str] = EmbeddingMethod.SBERT,
    cloned_repo_base_dir: str = str(Path.cwd()),
    should_reindex: bool = True,
    should_push_to_repo: bool = True,
) -> RepoIndex:
    """
    1. Clone or pull `repo_url`
    2. Embed each file into a Chroma collection in `.kno/`
    3. Optionally commit & push the `.kno/` folder back to `repo_url`.
    """
    repo_path = clone_repo(repo_url, branch, cloned_repo_base_dir)
    
    if not should_reindex:
        # 2) locate .kno and filter for this embedding method
        kno_root = Path(repo_path) / ".kno"
        if not kno_root.exists():
            raise FileNotFoundError(
                f"No .kno directory in {repo_path}. Run clone_and_index first."
            )

        # Handle string input for embedding
        if isinstance(embedding, str):
            try:
                embedding = EmbeddingMethod(embedding)
            except ValueError:
                raise ValueError(f"Invalid embedding method: {embedding}. Must be one of {[e.value for e in EmbeddingMethod]}")

        prefix = f"embedding_{embedding.value}_"
        cand_dirs = [
            d for d in kno_root.iterdir() if d.is_dir() and d.name.startswith(prefix)
        ]
        if not cand_dirs:
            raise ValueError(
                f"No embedding folders for `{embedding.value}` found in {kno_root}"
            )

        latest_dir = max(cand_dirs, key=_ts)
        embed_fn = (
            OpenAIEmbeddings()
            if embedding.value == "OpenAIEmbedding"
            else SBERTEmbeddings()
        )
        vs = Chroma(
            collection_name=repo_path.name,
            embedding_function=embed_fn,
            persist_directory=str(latest_dir),
        )
        skip_dirs = {".git", "node_modules", "build", "dist", "target", ".vscode", ".kno"}
        skip_files = {"package-lock.json", "yarn.lock", ".prettierignore"}
        digest = _build_directory_digest(repo_path, skip_dirs, skip_files)
        return RepoIndex(vector_store=vs, digest=digest, path=repo_path)
    
    repo_index = index_repo(repo_path, embedding)
    if should_push_to_repo:
        push_to_repo(repo_path)
    return repo_index


chroma_vs = None

def search(
    repo_url: str,
    branch: str = "main",
    embedding: EmbeddingMethod = EmbeddingMethod.SBERT,
    query: str = "",
    k: int = 8,
    cloned_repo_base_dir: str = str(Path.cwd()),
) -> List[str]:
    """
    1. Clone/pull `repo_url`
    2. Load the existing `.kno/` Chroma DB
    3. Return the top‐k page_content for `query`
    """
    global chroma_vs
    if not chroma_vs:
        repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
        repo_path = os.path.join(cloned_repo_base_dir, repo_name)

        if not Path(repo_path).exists():
            Repo.clone_from(repo_url, repo_path, depth=1, branch=branch)
        else:
            Repo(repo_path).remotes.origin.pull(branch)

        # 2) locate .kno and filter for this embedding method
        kno_root = Path(repo_path) / ".kno"
        if not kno_root.exists():
            raise FileNotFoundError(
                f"No .kno directory in {repo_path}. Run clone_and_index first."
            )

        prefix = f"embedding_{embedding.value}_"
        cand_dirs = [
            d for d in kno_root.iterdir() if d.is_dir() and d.name.startswith(prefix)
        ]
        if not cand_dirs:
            raise ValueError(
                f"No embedding folders for `{embedding.value}` found in {kno_root}"
            )

        latest_dir = max(cand_dirs, key=_ts)
        embed_fn = (
            OpenAIEmbeddings()
            if embedding.value == "OpenAIEmbedding"
            else SBERTEmbeddings()
        )
        chroma_vs = Chroma(
            collection_name=repo_name,
            embedding_function=embed_fn,
            persist_directory=str(latest_dir),
        )

    return [d.page_content for d in chroma_vs.similarity_search(query, k=k)]


def load_index(
    repo_path: Path,
    embedding: Union[EmbeddingMethod, str] = EmbeddingMethod.SBERT,
) -> RepoIndex:
    """
    Load an existing index from a directory containing a .kno folder.
    
    Args:
        repo_path: Path to the repository directory containing .kno folder
        embedding: EmbeddingMethod.OPENAI or EmbeddingMethod.SBERT, or their string values
        
    Returns:
        RepoIndex object with the vector store and repository information
        
    Raises:
        FileNotFoundError: If .kno directory doesn't exist
        ValueError: If no embedding folders found for the specified embedding method
    """
    # Handle string input for embedding
    if isinstance(embedding, str):
        try:
            embedding = EmbeddingMethod(embedding)
        except ValueError:
            raise ValueError(f"Invalid embedding method: {embedding}. Must be one of {[e.value for e in EmbeddingMethod]}")

    # Locate .kno and filter for this embedding method
    kno_root = Path(repo_path) / ".kno"
    if not kno_root.exists():
        raise FileNotFoundError(
            f"No .kno directory in {repo_path}. Run clone_and_index first."
        )

    prefix = f"embedding_{embedding.value}_"
    cand_dirs = [
        d for d in kno_root.iterdir() if d.is_dir() and d.name.startswith(prefix)
    ]
    if not cand_dirs:
        raise ValueError(
            f"No embedding folders for `{embedding.value}` found in {kno_root}"
        )

    latest_dir = max(cand_dirs, key=_ts)
    embed_fn = (
        OpenAIEmbeddings()
        if embedding.value == "OpenAIEmbedding"
        else SBERTEmbeddings()
    )
    
    vs = Chroma(
        collection_name=repo_path.name,
        embedding_function=embed_fn,
        persist_directory=str(latest_dir),
    )
    
    skip_dirs = {".git", "node_modules", "build", "dist", "target", ".vscode", ".kno"}
    skip_files = {"package-lock.json", "yarn.lock", ".prettierignore"}
    digest = _build_directory_digest(repo_path, skip_dirs, skip_files)
    
    return RepoIndex(vector_store=vs, digest=digest, path=repo_path)
