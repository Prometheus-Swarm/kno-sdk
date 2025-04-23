import logging
import pathlib

from pathlib import Path
from git import Repo
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from chromadb import Client as Chroma
from typing import Dict, List, Tuple
from tree_sitter_languages import get_language
from tree_sitter import Parser

Language = get_language
logger = logging.getLogger(__name__)
TOKEN_LIMIT = 16_000                        # per-chunk token cap

INDEX_CHUNK_LIMIT = 400

LANG_NODE_TARGETS: Dict[str, Tuple[str, ...]] = {
    "python": ("function_definition", "class_definition"),
    "javascript": ("function", "method_definition", "class"),
    "typescript": ("function", "method_definition", "class"),
    "java":      ("method_declaration", "class_declaration", "interface_declaration"),
    "go":        ("function_declaration", "method_declaration", "type_specifier"),
    "c":         ("function_definition",),
    "cpp":       ("function_definition", "class_specifier", "struct_specifier"),
    "rust":      ("function_item", "struct_item", "enum_item", "mod_item"),
    "php":       ("function_definition", "class_declaration"),
    "ruby":      ("method", "class", "module"),
    "kotlin":    ("function_declaration", "class_declaration", "object_declaration"),
}
EXT_TO_LANG = {
    # Python
    ".py":   "python",
    ".pyi":  "python",
    # JavaScript
    ".js":   "javascript",
    ".jsx":  "javascript",
    ".mjs":  "javascript",
    ".cjs":  "javascript",
    # TypeScript
    ".ts":   "typescript",
    ".tsx":  "typescript",
    # Java
    ".java": "java",
    # Go
    ".go":   "go",
    # C
    ".c":    "c",
    ".h":    "c",
    # C++
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".cxx":  "cpp",
    ".hpp":  "cpp",
    ".hh":   "cpp",
    ".hxx":  "cpp",
    # Rust
    ".rs":   "rust",
    # PHP
    ".php":  "php",
    # Ruby
    ".rb":   "ruby",
    # Kotlin
    ".kt":   "kotlin",
    ".kts":  "kotlin",
}

# extensions that are almost always binary blobs
BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp",
    ".ico", ".tiff", ".mp3", ".wav", ".ogg", ".flac",
    ".mp4", ".mkv", ".mov", ".avi", ".wmv",
    ".pdf", ".psd", ".ai", ".eps", ".ttf", ".otf", ".woff", ".woff2",
    ".zip", ".gz", ".tar", ".7z", ".rar", ".exe", ".msi", ".dll",
    ".mp4", ".mkv", ".mov", ".avi", ".wmv",
}
# ─────────── Load grammars (works w/ tree_sitter 0.20 → 0.22) ────────────
LANGUAGE_CACHE: Dict[str, Language] = {}
for lang_name in set(EXT_TO_LANG.values()):
    try:
        LANGUAGE_CACHE[lang_name] = Language(lang_name)
    except TypeError:
        logger.warning("No grammar for %s (%s) – falling back to line chunking", lang_name, exc)


PARSER_CACHE: Dict[str, Parser] = {
    lang: (lambda l: (p := Parser(), p.set_language(l), p)[0])(lang_obj)
    for lang, lang_obj in LANGUAGE_CACHE.items()
}


class RepoIndex:
    path: Path
    vector_store: Chroma
    digest: str

    def _build_directory_digest(
        repo_path: pathlib.Path, skip_dirs: set[str], skip_files: set[str]
    ) -> str:
        lines: List[str] = []
        for root, dirs, files in os.walk(repo_path):
            rel_root = pathlib.Path(root).relative_to(repo_path)
            if any(p in skip_dirs for p in rel_root.parts):
                dirs.clear()
                continue
            files = [f for f in files if f not in skip_files]
            if not files:
                continue
            depth = len(rel_root.parts)
            indent = "    " * depth
            dir_display = "." if rel_root == pathlib.Path(".") else f"{rel_root}/"
            lines.append(f"{indent}{dir_display} ( {len(files)} files )")
            for f in files:
                lines.append(f"{indent}    {f}")
            if sum(len(l) for l in lines) > 4000:  # ≈1 k tokens
                lines.append("…")
                break
        return "\n".join(lines)

def _extract_semantic_chunks(path: pathlib.Path, text: str) -> List[str]:
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

def _fallback_line_chunks(text: str) -> List[str]:
    lines = text.splitlines()
    return [
        "\n".join(lines[i : i + INDEX_CHUNK_LIMIT])
        for i in range(0, len(lines), INDEX_CHUNK_LIMIT)
    ]

class SBERTEmbeddings(Embeddings):
    def __init__(self, model_name: str = "microsoft/graphcodebert-base"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=True).tolist()
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

def clone_and_index(
    repo_url: str,
    branch: str = "main",
    embedding: str = "openai",
    base_dir: Path = Path.cwd()
) -> RepoIndex:
    """
    1. Clone or pull `repo_url`
    2. Embed each file into a Chroma collection in `.kno/`
    3. Commit & push the `.kno/` folder back to `repo_url`.
    """
    repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    repo_path = base_dir / repo_name
    kno_dir = repo_path / ".kno"

    # 1. clone or pull
    if not repo_path.exists():
        logger.info("Cloning %s → %s", repo_url, repo_path)
        Repo.clone_from(repo_url, repo_path, depth=1, branch=branch)
    else:
        logger.info("Pulling latest on %s", repo_name)
        Repo(repo_path).remotes.origin.pull(branch)

    # 2. choose embedding
    embed_fn = OpenAIEmbeddings() if embedding=="openai" else SBERTEmbeddings()
    chroma_client = Chroma(persist_directory=str(kno_dir))
    vs = chroma_client.get_or_create_collection(name=repo_name, embedding_function=embed_fn)

    # 3. index if empty
    if vs.count() == 0:
        logger.info("Indexing %s …", repo_name)
        texts, metas = [], []
        skip_dirs = {".git","node_modules","build","dist","target",".vscode"}
        skip_files= {"package-lock.json","yarn.lock",".prettierignore"}
        digest = _build_directory_digest(repo_path, skip_dirs, skip_files)

        for fp in repo_path.rglob("*.*"):
            if any(p in skip_dirs for p in fp.parts) or fp.name in skip_files:
                continue
            if fp.stat().st_size>2_000_000 or fp.suffix.lower() in BINARY_EXTS:
                continue
            content = fp.read_text(errors="ignore")
            chunks = _extract_semantic_chunks(fp,content) or _fallback_line_chunks(content)
            for chunk in chunks:
                texts.append(chunk[:TOKEN_LIMIT])
                metas.append({"source": str(fp.relative_to(repo_path))})
        vs.add(documents=texts, metadatas=metas)
        logger.info("Embedded %d chunks", len(texts))

    # 4. commit & push .kno
    git = Repo(repo_path)
    git.git.add(str(kno_dir))
    git.index.commit("Add/update .kno embedding database")
    git.remote().push(branch)

    return RepoIndex(vs, digest)

def search(
    repo_url: str,
    branch: str = "main",
    embedding: str = "openai",
    query: str = "",
    k: int = 8,
    base_dir: Path = Path.cwd()
) -> List[str]:
    """
    1. Clone/pull `repo_url`
    2. Load the existing `.kno/` Chroma DB
    3. Return the top‐k page_content for `query`
    """
    repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    repo_path = base_dir / repo_name
    kno_dir = repo_path / ".kno"

    if not repo_path.exists():
        Repo.clone_from(repo_url, repo_path, depth=1, branch=branch)
    else:
        Repo(repo_path).remotes.origin.pull(branch)

    embed_fn = OpenAIEmbeddings() if embedding=="openai" else SBERTEmbeddings()
    chroma_client = Chroma(persist_directory=str(kno_dir))
    vs = chroma_client.get_collection(name=repo_name, embedding_function=embed_fn)

    docs = vs.similarity_search(query, n_results=k)
    return [d.page_content for d in docs]
