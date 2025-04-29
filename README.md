# kno-sdk

A Python library for cloning, indexing, and semantically searching Git repositories using embeddings (OpenAI or SBERT) and Chroma — plus a high-level `agent_query` for autonomous code agent.

---

## 🚀 Features

- **Clone or update** any Git repository with a single call  
- **Extract semantic code chunks** via Tree-Sitter grammars (functions, classes, methods, etc.)  
- **Fallback to line-based chunking** for unsupported languages or large files  
- **Embed code or text** with your choice of:
  - OpenAI's `text-embedding-ada-002` via **OpenAIEmbeddings**  
  - Local SBERT model (e.g. `microsoft/graphcodebert-base`) via **SBERTEmbeddings**  
- **Persist vector store** in a `.kno/` folder using Chroma  
- **Auto-commit & push** the embedding database back to your repo  
- **Fast similarity search** over indexed code chunks  
- **Autonomous agent** for code analysis via `agent_query()`

---

## 📦 Installation

```bash
pip install kno-sdk
```

🏁 Quickstart
-------------

```python
from kno_sdk import clone_and_index, search, EmbeddingMethod

# 1. Clone (or pull) and index a repository
repo_index = clone_and_index(
    repo_url="https://github.com/SyedGhazanferAnwar/NestJs-MovieApp",
    branch="master",
    embedding=EmbeddingMethod.SBERT,      # or EmbeddingMethod.OPENAI
    cloned_repo_base_dir="repos"                      # where to clone locally
)
print("Indexed at:", repo_index.path)
print("Directory snapshot:\n", repo_index.digest)

# 2. Perform semantic search
results = search(
    repo_url="https://github.com/SyedGhazanferAnwar/NestJs-MovieApp",
    branch="master",
    embedding=EmbeddingMethod.SBERT,
    cloned_repo_base_dir="repos",
    query="NestFactory",
    k=5
)
for i, chunk in enumerate(results, 1):
    print(f"--- Result #{i} ---\n{chunk}\n")

# 3. Autonomous Code-Analysis Agent
from kno_sdk import agent_query, EmbeddingMethod, LLMProvider

# First create a repo index
repo_index = clone_and_index(
    repo_url="https://github.com/WebGoat/WebGoat",
    branch="main",
    embedding=EmbeddingMethod.SBERT,
    cloned_repo_base_dir="repos"
)

# Then use the index with agent_query
result = agent_query(
    repo_index=repo_index,
    llm_provider=LLMProvider.ANTHROPIC,
    llm_model="claude-3-haiku-20240307",
    llm_temperature=0.0,
    llm_max_tokens=4096,
    llm_system_prompt="You are a senior code-analysis agent.",
    prompt="Find issues, bugs and vulnerabilities in this repo, and explain each with exact code locations.",
    MODEL_API_KEY="your_api_key_here"
)

print(result)
```



📖 API Reference
----------------

### clone\_and\_index(...) → RepoIndex

Clone (or pull) a repository, embed its files, and persist a Chroma database in .kno folder. Finally, commit & push the .kno/ folder back to the original repo.

```python
def clone_and_index(
    repo_url: str,
    branch: str = "main",
    embedding: EmbeddingMethod = EmbeddingMethod.SBERT,
    cloned_repo_base_dir: str = "."
) -> RepoIndex
```

*   **repo\_url** — Git HTTPS/SSH URL
    
*   **branch** — branch to clone or update (default: main)
    
*   **embedding** — EmbeddingMethod.OPENAI or EmbeddingMethod.SBERT
    
*   **base\_dir** — local directory to clone into (default: current working dir)
    

Returns a `RepoIndex` object with:

*   path: pathlib.Path — local clone directory
    
*   digest: str — textual snapshot of the directory tree
    
*   vector\_store: Chroma — the Chroma collection instance
    

### search(...) → List[str]

Run a similarity search on an existing `.kno/` Chroma database.

```python
def search(
    repo_url: str,
    branch: str = "main",
    embedding: EmbeddingMethod = EmbeddingMethod.SBERT,
    query: str = "",
    k: int = 8,
    cloned_repo_base_dir: str = "."
) -> List[str]
```

*   **query** — your natural-language or code search prompt
    
*   **k** — number of top results to return
    

Returns a list of the top-k matching code/text chunks.

### agent_query(...) → str

High-level agent that clones, indexes, and then iteratively uses tools (search_code, read_file, etc.) plus an LLM to fulfill your prompt.

```python
def agent_query(
    repo_url: str,
    branch: str = "main",
    embedding: EmbeddingMethod = EmbeddingMethod.SBERT,
    cloned_repo_base_dir: str = str(Path.cwd()),
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC,
    llm_model: str = "claude-3-haiku-20240307",
    llm_temperature: float = 0.0,
    llm_max_tokens: int = 4096,
    llm_system_prompt: str = "",
    prompt: str = "",
    MODEL_API_KEY: str = "",
) -> str
```
*   **repo\_url**, **branch**, **embedding**, **base\_dir** — same as above
    
*   **llm\_provider** — LLMProvider.OPENAI or LLMProvider.ANTHROPIC
    
*   **llm\_model** — model name (e.g. "gpt-4" or "claude-3-haiku-20240307")
    
*   **llm\_temperature**, **llm\_max\_tokens** — sampling params
    
*   **llm\_system\_prompt** — initial system message for the agent
    
*   **prompt** — your user query/task description
    
*   **MODEL\_API\_KEY** — sets OPENAI\_API\_KEY or ANTHROPIC\_API\_KEY
    

Returns the agent's **Final Answer** as a string.

### EmbeddingMethod

```python

class EmbeddingMethod(str, Enum):
    OPENAI = "OpenAIEmbeddings"
    SBERT  = "SBERTEmbeddings"
```

Choose between OpenAI's hosted embeddings or a local SBERT model.

🔍 How It Works
---------------

1.  **Clone or Pull**Uses GitPython to clone depth-1 or pull the latest changes.
    
2.  **Directory Snapshot**Builds a small "digest" of files/folders (up to ~1 K tokens).
    
3.  **Chunk Extraction**
    
    *   **Tree-sitter** for language-aware extraction of functions, classes, etc.
        
    *   **Fallback** to fixed-size line chunks for unknown languages or large files.
        
4.  **Embedding**
    
    *   Streams each chunk into your chosen embedding backend.
        
    *   Respects a 16 000-token cap per chunk.
        
5.  **Vector Store**
    
    *   Persists embeddings in a namespaced Chroma collection under .kno/.
        
    *   Only indexes files once (skips already-populated collections).
        
6.  **Commit & Push**
    
    *   Automatically stages, commits, and pushes .kno/ back to your remote.
        
7.   **Autonomous Agent**
    
*   RAG prompt
    
*   Tool calls (search\_code, read\_file, …)
    
*   Iterative LLM planning & execution
    
*   Stops on "Final Answer:" or max iterations
  
⚙️ Configuration
----------------

*   **Skip directories**: .git, node\_modules, build, dist, target, .vscode, .kno
    
*   **Skip files**: package-lock.json, yarn.lock, .prettierignore
    
*   **Binary extensions**: common image, audio, video, archive, font, and binary file types
    

All of the above can be modified by forking the source and adjusting the `skip_dirs`, `skip_files`, and `BINARY_EXTS` sets.

🔧 Dependencies
---------------

*   [GitPython](https://pypi.org/project/GitPython/)
    
*   [langchain-openai](https://pypi.org/project/langchain-openai/)
    
*   [sentence-transformers](https://pypi.org/project/sentence-transformers/)
    
*   [langchain-chroma](https://pypi.org/project/langchain-chroma/)
    
*   [tree-sitter-languages](https://pypi.org/project/tree-sitter-languages/)
    
*   [tree-sitter](https://pypi.org/project/tree-sitter/)
    

🤝 Contributing
---------------

1.  Fork this repo
    
2.  Create your feature branch (git checkout -b feature/AmazingFeature)
    
3.  Commit your changes (git commit -m 'Add amazing feature')
    
4.  Push to the branch (git push origin feature/AmazingFeature)
    
5.  Open a Pull Request
    

Please run pytest before submitting and follow the existing code style.