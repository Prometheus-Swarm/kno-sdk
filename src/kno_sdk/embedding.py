from typing import Dict, List, Tuple, Optional, Any, TypedDict
from tree_sitter_languages import get_language
from tree_sitter import Parser
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from enum import Enum
from git import Repo

import logging
from pathlib import Path
import os
import time


Language = get_language
logger = logging.getLogger(__name__)

MAX_FALLBACK_LINES = 150
TOKEN_LIMIT = 16_000  # per-chunk token cap

LANG_NODE_TARGETS: Dict[str, Tuple[str, ...]] = {
    "python": ("function_definition", "class_definition"),
    "javascript": ("function", "method_definition", "class"),
    "typescript": ("function", "method_definition", "class"),
    "java": ("method_declaration", "class_declaration", "interface_declaration"),
    "go": ("function_declaration", "method_declaration", "type_specifier"),
    "c": ("function_definition",),
    "cpp": ("function_definition", "class_specifier", "struct_specifier"),
    "rust": ("function_item", "struct_item", "enum_item", "mod_item"),
    "php": ("function_definition", "class_declaration"),
    "ruby": ("method", "class", "module"),
    "kotlin": ("function_declaration", "class_declaration", "object_declaration"),
}
EXT_TO_LANG = {
    # Python
    ".py": "python",
    ".pyi": "python",
    # JavaScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".json": "javascript",
    # TypeScript
    ".ts": "typescript",
    ".tsx": "typescript",
    # Java
    ".java": "java",
    # Go
    ".go": "go",
    # C
    ".c": "c",
    ".h": "c",
    # C++
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    # Rust
    ".rs": "rust",
    # PHP
    ".php": "php",
    # Ruby
    ".rb": "ruby",
    # Kotlin
    ".kt": "kotlin",
    ".kts": "kotlin",
}

# extensions that are almost always binary blobs
BINARY_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".svg",
    ".webp",
    ".ico",
    ".tiff",
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".wmv",
    ".pdf",
    ".psd",
    ".ai",
    ".eps",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".zip",
    ".gz",
    ".tar",
    ".7z",
    ".rar",
    ".exe",
    ".msi",
    ".dll",
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".wmv",
}

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


class EmbeddingMethod(str, Enum):
    OPENAI = "OpenAIEmbedding"
    SBERT = "SBERTEmbedding"


class RepoIndex:
    path: Path
    vector_store: Chroma
    digest: str

    def __init__(self, vector_store: Chroma, digest: str, path: Path = Path.cwd()):
        self.path = path
        self.vector_store = vector_store
        self.digest = digest

    def _build_directory_digest(
        repo_path: Path, skip_dirs: set[str], skip_files: set[str]
    ) -> str:
        lines: List[str] = []
        for root, dirs, files in os.walk(repo_path):
            rel_root = Path(root).relative_to(repo_path)
            if any(p in skip_dirs for p in rel_root.parts):
                dirs.clear()
                continue
            files = [f for f in files if f not in skip_files]
            if not files:
                continue
            depth = len(rel_root.parts)
            indent = "    " * depth
            dir_display = "." if rel_root == Path(".") else f"{rel_root}/"
            lines.append(f"{indent}{dir_display} ( {len(files)} files )")
            for f in files:
                lines.append(f"{indent}    {f}")
            if sum(len(l) for l in lines) > 4000:  # ≈1 k tokens
                lines.append("…")
                break
        return "\n".join(lines)


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


def _build_directory_digest(
    repo_path: Path, skip_dirs: set[str], skip_files: set[str]
) -> str:
    lines: List[str] = []
    for root, dirs, files in os.walk(repo_path):
        rel_root = Path(root).relative_to(repo_path)
        if any(p in skip_dirs for p in rel_root.parts):
            dirs.clear()
            continue
        files = [f for f in files if f not in skip_files]
        if not files:
            continue
        depth = len(rel_root.parts)
        indent = "    " * depth
        dir_display = "." if rel_root == Path(".") else f"{rel_root}/"
        lines.append(f"{indent}{dir_display} ( {len(files)} files )")
        for f in files:
            lines.append(f"{indent}    {f}")
        if sum(len(l) for l in lines) > 4000:  # ≈1 k tokens
            lines.append("…")
            break
    return "\n".join(lines)


# 3) parse out the timestamp and pick the max
def _ts(d: Path) -> int:
    parts = d.name.split("_")
    try:
        return int(parts[2])
    except (IndexError, ValueError):
        return 0


def clone_and_index(
    repo_url: str,
    branch: str = "main",
    embedding: EmbeddingMethod = EmbeddingMethod.SBERT,
    cloned_repo_base_dir: str = str(Path.cwd()),
    should_reindex: bool = True,
    should_push_to_repo: bool = True,
) -> RepoIndex:
    """
    1. Clone or pull `repo_url`
    2. Embed each file into a Chroma collection in `.kno/`
    3. Commit & push the `.kno/` folder back to `repo_url`.
    """
    if isinstance(embedding, str):
        embedding = EmbeddingMethod(embedding)  # raises ValueError if invalid

    repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    repo_path = os.path.join(cloned_repo_base_dir, repo_name)
    kno_dir = os.path.join(repo_path, ".kno")
    skip_dirs = {".git", "node_modules", "build", "dist", "target", ".vscode", ".kno"}
    skip_files = {"package-lock.json", "yarn.lock", ".prettierignore"}
    digest = _build_directory_digest(repo_path, skip_dirs, skip_files)

    # 1. clone or pull
    if not Path(repo_path).exists():
        logger.info("Cloning %s → %s", repo_url, repo_path)
        Repo.clone_from(repo_url, repo_path, depth=1, branch=branch)
    else:
        logger.info("Pulling latest on %s", repo_name)
        Repo(repo_path).remotes.origin.pull(branch)
    if not should_reindex:
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
        vs = Chroma(
            collection_name=repo_name,
            embedding_function=embed_fn,
            persist_directory=str(latest_dir),
        )
        return RepoIndex(vector_store=vs, digest=digest, path=Path(repo_path))

    repo = Repo(repo_path)
    commit = repo.head.commit.hexsha[:7]
    time_ms = int(time.time() * 1000)
    subdir = f"embedding_{embedding.value}_{time_ms}_{commit}"
    # 2. choose embedding
    embed_fn = (
        OpenAIEmbeddings()
        if embedding.value == "OpenAIEmbedding"
        else SBERTEmbeddings()
    )
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

    # 4. commit & push .kno
    # get path relative to repo root:
    if should_push_to_repo:
        try:
            logger.info("Pushing .kno to %s", repo_url)
            relative_kno = os.path.relpath(str(kno_dir), str(repo_path))
            repo.git.add(str(relative_kno))
            repo.index.commit("Add/update .kno embedding database")
            repo.remote().push(branch)
        except Exception as e:
            logger.warning("Failed to push .kno to %s: %s", repo_url, e)

    return RepoIndex(vector_store=vs, digest=digest, path=Path(repo_path))


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
