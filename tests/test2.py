from src.kno_sdk import clone_and_index, EmbeddingMethod, agent_query, search

# from kno_sdk import clone_and_index, search, EmbeddingMethod, agent_query
import os
from dotenv import load_dotenv

load_dotenv()
# Forking
# x = clone_and_index("https://github.com/gothinkster/node-express-realworld-example-app", branch="master", embedding=EmbeddingMethod.SBERT, cloned_repo_base_dir="repos",should_push_to_repo=False)
# print(x)

repo_url = "https://github.com/SyedGhazanferAnwar/NestJs-MovieApp"
branch = "master"
system_prompt = f"""
You are a senior code-analysis agent working **on the repository below**.

Repository: {repo_url}  
Branch:     {branch}

Your tasks, in order:

1. **Answer the user's request.**
2. If you lack information, decide which tool to use to get it.  
   • `read_file` – to read code or config files  
   • `search_code` – semantic search across the repo  

Keep using the tools until you have enough information. Once ready, reply with:

---

Final Answer:

```json
{{
  "name": "example-project",
  "description": "A cross-platform desktop application for note-taking and task management.",
  "repository_url": "https://github.com/username/example-project",
  "technical_metadata": {{
    "primary_language": "C++",
    "languages_used": [
      {{"language": "C++", "percentage": 85.0}},
      {{"language": "QML", "percentage": 10.0}},
      {{"language": "Shell", "percentage": 5.0}}
    ],
    "frameworks": [
      {{"name": "Qt", "version": "6.5"}},
      {{"name": "Boost", "version": "1.81"}}
    ],
    "build_tools": [
      {{"name": "CMake", "version": "3.27"}},
      {{"name": "Make"}}
    ],
    "test_frameworks": [
      {{"name": "Catch2", "version": "3.3"}}
    ],
    "linters": [
      {{"name": "clang-tidy"}},
      {{"name": "cppcheck"}}
    ],
    "ci_cd": {{
      "tools": ["GitHub Actions"],
      "config_files": [".github/workflows/build.yml"]
    }},
    "packaging": {{
      "method": "CMake + CPack",
      "output_formats": [".tar.gz", ".deb"]
    }},
    "deployment": {{
      "type": "desktop",
      "platforms": ["Linux", "Windows", "macOS"]
    }}
  }},
  "functionality": {{
    "application_type": "Desktop",
    "core_features": [
      "Note editing and formatting",
      "Task tagging and reminders",
      "Sync with local filesystem"
    ],
    "authentication": {{
      "used": false
    }},
    "data_storage": {{
      "type": "Local",
      "format": "SQLite database",
      "data_models": 7
    }},
    "external_dependencies": [
      {{"name": "sqlite", "version": "3.39"}},
      {{"name": "zlib", "version": "1.2.13"}}
    ]
  }}
}}
"""
prompt = """
        Before making any changes, can you summarize the architecture and key components of this GitHub repo as you understand it from the current context? 
        Please include the main technologies used, key folders/files, and the primary functionality implemented by reading all the important files.
        If you are missing any crucial files or information, mention that too.
    """
resp = agent_query(
    repo_url=repo_url,
    branch=branch,
    embedding=EmbeddingMethod.SBERT,
    cloned_repo_base_dir="repos",
    llm_system_prompt=system_prompt,
    prompt=prompt,
    MODEL_API_KEY=os.environ.get("ANTHROPIC_API_KEY"),
)
print(resp)

# y = search("https://github.com/SyedGhazanferAnwar/NestJs-MovieApp", branch="master", embedding=EmbeddingMethod.SBERT, cloned_repo_base_dir="repos",query="NEST")
# print(y)

# Search coming empty everything else ready


# Models to use:
# jinaai/jina-embeddings-v2-base-code
# microsoft/graphcodebert-base



# ISSUE remaining, the chunking might be very bad as the search file always returns empty / bad output