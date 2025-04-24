import logging
import pathlib
import os
import time
import json
import re


from pathlib import Path
from git import Repo
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from typing import Dict, List, Tuple, Optional, Any, TypedDict
from tree_sitter_languages import get_language
from tree_sitter import Parser
from enum import Enum

from dataclasses import dataclass
from langchain_anthropic import ChatAnthropic
from langchain.chat_models.base import BaseChatModel
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END


Language = get_language
logger = logging.getLogger(__name__)
TOKEN_LIMIT = 16_000  # per-chunk token cap
MAX_ITERATIONS = 30
INDEX_CHUNK_LIMIT = 400

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
        logger.warning(
            "No grammar for %s (%s) – falling back to line chunking", lang_name, exc
        )


PARSER_CACHE: Dict[str, Parser] = {
    lang: (lambda l: (p := Parser(), p.set_language(l), p)[0])(lang_obj)
    for lang, lang_obj in LANGUAGE_CACHE.items()
}


class EmbeddingMethod(str, Enum):
    OPENAI = "OpenAIEmbedding"
    SBERT = "SBERTEmbedding"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# ──────────────────────────── AGENT FACTORY ──────────────────────────
@dataclass
class AgentConfig:
    repo_url: str
    branch: str = "main"
    llm_provider: str = "anthropic"
    model_name: str = "claude-3-haiku-20240307"
    temperature: float = 0.0
    embedding_function: str = "SBERTEmbedding"
    max_tokens: int = 4096


# ─────────────────────────── LLM PROVIDERS ────────────────────────────
class LLMProviderBase(BaseChatModel):
    provider_name: str = "abstract"

    @property
    def _llm_type(self) -> str:
        return self.provider_name


class OpenAIProvider(ChatOpenAI, LLMProviderBase):
    provider_name: str = "openai"


class AnthropicProvider(ChatAnthropic, LLMProviderBase):
    provider_name: str = "anthropic"


class RepoIndex:
    path: Path
    vector_store: Chroma
    digest: str

    def __init__(
        self, vector_store: Chroma, digest: str, path: pathlib.Path = pathlib.Path.cwd()
    ):
        self.path = path
        self.vector_store = vector_store
        self.digest = digest

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


# ───────────────────────────── TOOLS ────────────────────────────────


def build_tools(index: RepoIndex, llm: LLMProviderBase) -> List[Tool]:
    """Return lightweight LangChain `Tool`s that close over *index*."""

    def search_code(query: str, k: int = 8) -> str:
        if not query or query.strip() == "":
            # Return directory structure instead of empty search
            return (
                f"Please provide a search query. Repository structure:\n{index.digest}"
            )
        # 1) retrieve top‑k code snippets
        snippets = search(query, k=k)
        context = "\n\n---\n\n".join(snippets)

        # 2) build a RAG prompt
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a senior code‑analysis assistant for repository "
                    f"'{index.path.name}'."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Here are the top relevant code snippets:\n\n"
                    f"{context}\n\n"
                    "Using *only* the above, answer the question:\n\n"
                    f"{query}"
                ),
            },
        ]

        # 3) invoke the LLM to generate
        response = llm.invoke(messages)
        return response.content

    def read_file(
        file_path: str, start: Optional[int] = None, end: Optional[int] = None
    ) -> str:
        try:
            text = (index.path / file_path).read_text(errors="ignore")
            if start is not None or end is not None:
                text = "\n".join(text.splitlines()[start:end])
            return text[:TOKEN_LIMIT]
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"

    return [
        Tool(name="search_code", func=search_code, description="Semantic code search"),
        Tool(name="read_file", func=read_file, description="Read file content"),
    ]


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


# ────────────────────── LANGGRAPH STATE AND NODES ───────────────────────


class AgentState(TypedDict):
    input: str
    repo_info: Dict
    messages: List[Dict[str, Any]]
    intermediate_steps: List[tuple]
    iterations: int


def create_agent_graph(tools: List[Tool], llm: LLMProviderBase, system_message: str):
    """Create a LangGraph agent with the provided tools and LLM."""

    # Function to formulate prompt with history and tool results
    def get_prompt_with_history(state: AgentState) -> List[Dict[str, Any]]:
        messages = []

        # Add system message
        messages.append({"role": "system", "content": system_message})

        # Initial human input
        messages.append({"role": "user", "content": state["input"]})

        # Add all intermediate steps as messages
        for i, (action, observation) in enumerate(state["intermediate_steps"]):
            # Action message - what tool was used and with what input
            if isinstance(action, dict):
                tool_name = action.get("name", "unknown")
                tool_input = action.get("arguments", {})
                action_str = f"I'll use the {tool_name} tool with input: {json.dumps(tool_input, indent=2)}"
            else:
                action_str = f"I'll use the {action} tool"

            messages.append({"role": "assistant", "content": action_str})

            # Observation message - what was the result of the tool
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            messages.extend(state["messages"])
        return messages

    # Node 1: Agent thinks about what to do next
    def agent_thinking(state: AgentState) -> AgentState:
        messages = get_prompt_with_history(state)

        prompt_suffix = """
        Based on the above, decide on the next best step. You can:
        
        1. Use a tool to gather more information by formatting your response as:
        ```json
        {
          "action": "tool_name",
          "action_input": "input_value" or {"param1": "value1", "param2": "value2"}
        }
        ```
        
        2. Provide a final answer when you've completed the task:
        ```
        Final Answer: Your comprehensive analysis or solution here.
        ```
        """

        # Add prompt suffix to guide response format
        messages.append({"role": "user", "content": prompt_suffix})

        # Get response from LLM
        response = llm.invoke(messages)

        # Return updated state
        return {
            **state,
            "messages": state["messages"]
            + [{"role": "assistant", "content": response.content}],
        }

    # Node 2: Parse action and execute tool if needed
    def execute_tools(state: AgentState) -> AgentState:
        """
        • Parse the assistant’s most‑recent message for a tool call.
        • Accept either fenced‑JSON *or* the natural‑language pattern
          “I'll use the <tool> tool with input: …”.
        • Execute the tool, append the observation, and advance the loop.
        • If no valid tool call is detected, inject a system nudge so the
          agent retries instead of entering an endless loop.
        """
        last_message = state["messages"][-1]["content"] if state["messages"] else ""

        # ---------- exit early on final answer ----------
        if "Final Answer:" in last_message:
            return {**state, "iterations": state["iterations"] + 1}

        # ---------- 1. fenced‑JSON tool call ------------
        tool_call = None
        m = re.search(r"```json\s*(\{.*?\})\s*```", last_message, re.DOTALL)
        if m:
            try:
                tool_call = json.loads(m.group(1))
            except json.JSONDecodeError:
                tool_call = None

        # ---------- 2. natural‑language pattern ---------
        if tool_call is None:
            nl = re.match(
                r"I'?ll use the (\w+) tool with input:\s*(.+)", last_message, re.I
            )
            if nl:
                raw_input = nl.group(2).strip()
                # remove symmetric quotes if present
                if raw_input[:1] in {"'", '"'} and raw_input[:1] == raw_input[-1:]:
                    raw_input = raw_input[1:-1]
                tool_call = {"action": nl.group(1).strip(), "action_input": raw_input}

        # ---------- 3. unable to parse ------------------
        if tool_call is None:
            return {
                **state,
                "messages": state["messages"]
                + [
                    {
                        "role": "system",
                        "content": (
                            "I couldn’t recognise a tool call. "
                            "Reply either with valid\n"
                            '```json\n{ "action": "tool", "action_input": … }\n```\n'
                            "or finish with **Final Answer:**."
                        ),
                    }
                ],
                "iterations": state["iterations"] + 1,
            }

        tool_name = tool_call.get("action")
        tool_input = tool_call.get("action_input")

        # ───── 4. ***Duplicate‑call guard*** – skip if same as last one ─────
        if state["intermediate_steps"]:
            prev_action, _ = state["intermediate_steps"][-1]
            if (
                isinstance(prev_action, dict)
                and prev_action.get("name") == tool_name
                and prev_action.get("arguments") == tool_input
            ):
                warn_msg = (
                    f"You already ran `{tool_name}` with that exact input. "
                    "I'll skip that and you can choose another action or finish with **Final Answer:**."
                )
                # Append a system message (not a new tool step), and do not bump iterations
                return {
                    **state,
                    "messages": state["messages"]
                    + [{"role": "system", "content": warn_msg}],
                }

        # ---------- 5. execute the tool -----------------
        for tool in tools:
            if tool.name == tool_name:
                try:
                    result = (
                        tool.func(**tool_input)
                        if isinstance(tool_input, dict)
                        else tool.func(tool_input)
                    )
                except Exception as exc:
                    result = f"Error executing {tool_name}: {exc}"

                # record intermediary step & dialogue
                new_steps = state["intermediate_steps"] + [
                    ({"name": tool_name, "arguments": tool_input}, result)
                ]
                tool_msg = {
                    "role": "assistant",
                    "content": f"```json\n{json.dumps(tool_call, indent=2)}\n```",
                }
                obs_msg = {"role": "user", "content": f"Observation: {result}"}

                return {
                    **state,
                    "messages": state["messages"] + [tool_msg, obs_msg],
                    "intermediate_steps": new_steps,
                    "iterations": state["iterations"] + 1,
                }

        # ---------- 5. tool not found -------------------
        err_msg = f"Tool '{tool_name}' not found."
        return {
            **state,
            "messages": state["messages"]
            + [{"role": "user", "content": f"Observation: {err_msg}"}],
            "intermediate_steps": state["intermediate_steps"]
            + [({"name": "error", "arguments": tool_call}, err_msg)],
            "iterations": state["iterations"] + 1,
        }

    # Routing function to decide next steps
    def should_continue(state: AgentState) -> str:
        # Check for final answer
        last_message = state["messages"][-1]["content"] if state["messages"] else ""
        if "Final Answer:" in last_message:
            return "end"

        if state["iterations"] >= MAX_ITERATIONS:
            # force the model to wrap up instead of ending the graph
            state["messages"].append(
                {
                    "role": "user",
                    "content": "You have hit the step limit. Summarise now using 'Final Answer:'.",
                }
            )
            return "continue"
        # Continue the loop
        return "continue"

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_thinking)
    workflow.add_node("tools", execute_tools)

    # Add edges
    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")

    # Set entry point
    workflow.set_entry_point("agent")

    return workflow.compile()


class AgentFactory:
    def _get_llm(self, cfg: AgentConfig) -> LLMProviderBase:
        if cfg.llm_provider == "openai":
            return OpenAIProvider(
                model_name=cfg.model_name, temperature=cfg.temperature
            )
        elif cfg.llm_provider == "anthropic":
            return AnthropicProvider(
                model=cfg.model_name,
                temperature=cfg.temperature,
                max_tokens_to_sample=cfg.max_tokens,
            )
        raise ValueError(f"Unknown provider: {cfg.llm_provider}")

    def create_agent(
        self, cfg: AgentConfig, base_dir: str = str(Path.cwd()), system_prompt: str = ""
    ):
        index = clone_and_index(
            cfg.repo_url, cfg.branch, cfg.embedding_function, base_dir, False
        )
        llm = self._get_llm(cfg)
        tools = build_tools(index, llm)
        agent_graph = create_agent_graph(tools, llm, system_prompt)

        # Create a wrapper that mimics the AgentExecutor.run method
        class AgentGraphRunner:
            def __init__(self, graph):
                self.graph = graph

            def run(self, input_str: str):
                state = {
                    "input": input_str,
                    "repo_info": {
                        "url": cfg.repo_url,
                        "branch": cfg.branch,
                        "digest": index.digest,
                    },
                    "messages": [],
                    "intermediate_steps": [],
                    "iterations": 0,
                }
                while True:
                    state = self.graph.invoke(
                        state, {"recursion_limit": MAX_ITERATIONS * 2}
                    )  # one step
                    last = state["messages"][-1]["content"] if state["messages"] else ""
                    if "Final Answer:" in last or state["iterations"] >= MAX_ITERATIONS:
                        break

                match = re.search(r"Final Answer:(.*)", last, re.DOTALL)
                return match.group(1).strip() if match else last

        return AgentGraphRunner(agent_graph)
    
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
    base_dir: str = str(Path.cwd()),
    should_reindex: bool = True,
) -> RepoIndex:
    """
    1. Clone or pull `repo_url`
    2. Embed each file into a Chroma collection in `.kno/`
    3. Commit & push the `.kno/` folder back to `repo_url`.
    """
    if isinstance(embedding, str):
        embedding = EmbeddingMethod(embedding)  # raises ValueError if invalid

    repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    repo_path = os.path.join(base_dir, repo_name)
    kno_dir = os.path.join(repo_path, ".kno")
    skip_dirs = {".git", "node_modules", "build", "dist", "target", ".vscode", ".kno"}
    skip_files = {"package-lock.json", "yarn.lock", ".prettierignore"}
    digest = _build_directory_digest(repo_path, skip_dirs, skip_files)

    # 1. clone or pull
    if not pathlib.Path(repo_path).exists():
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
        return RepoIndex(vector_store=vs, digest=digest)

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

        for fp in pathlib.Path(repo_path).rglob("*.*"):
            if any(p in skip_dirs for p in fp.parts) or fp.name in skip_files:
                continue
            if fp.stat().st_size > 2_000_000 or fp.suffix.lower() in BINARY_EXTS:
                continue
            content = fp.read_text(errors="ignore")
            chunks = _extract_semantic_chunks(fp, content) or _fallback_line_chunks(
                content
            )
            for chunk in chunks:
                texts.append(chunk[:TOKEN_LIMIT])
                metas.append({"source": str(fp.relative_to(repo_path))})
        vs.add_texts(texts=texts, metadatas=metas)
        logger.info("Embedded %d chunks", len(texts))

    # 4. commit & push .kno
    # get path relative to repo root:
    relative_kno = os.path.relpath(str(kno_dir), str(repo_path))
    repo.git.add(str(relative_kno))
    repo.index.commit("Add/update .kno embedding database")
    repo.remote().push(branch)

    return RepoIndex(vector_store=vs, digest=digest)


def search(
    repo_url: str,
    branch: str = "main",
    embedding: EmbeddingMethod = EmbeddingMethod.SBERT,
    query: str = "",
    k: int = 8,
    base_dir: str = str(Path.cwd()),
) -> List[str]:
    """
    1. Clone/pull `repo_url`
    2. Load the existing `.kno/` Chroma DB
    3. Return the top‐k page_content for `query`
    """
    repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    repo_path = os.path.join(base_dir, repo_name)

    if not pathlib.Path(repo_path).exists():
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
    vs = Chroma(
        collection_name=repo_name,
        embedding_function=embed_fn,
        persist_directory=str(latest_dir),
    )

    return [d.page_content for d in vs.similarity_search(query, k=k)]


def agent_query(
    repo_url: str,
    branch: str = "main",
    embedding: EmbeddingMethod = EmbeddingMethod.SBERT,
    base_dir: str = str(Path.cwd()),
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC,
    llm_model: str = "claude-3-haiku-20240307",
    llm_temperature: float = 0.0,
    llm_max_tokens: int = 4096,
    llm_system_prompt: str = "",
    prompt: str = "",
    MODEL_API_KEY: str = "",
):
    if LLMProvider.ANTHROPIC:
        os.environ["ANTHROPIC_API_KEY"] = MODEL_API_KEY
    elif LLMProvider.OPENAI:
        os.environ["OPENAI_API_KEY"] = MODEL_API_KEY
    cfg = AgentConfig(
        repo_url=repo_url,
        branch=branch,
        llm_provider=llm_provider.value,
        model_name=llm_model,
        embedding_function=embedding.value,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
    )
    agent = AgentFactory().create_agent(
        cfg, base_dir=base_dir, system_prompt=llm_system_prompt
    )
    result = agent.run(prompt)
    return result
