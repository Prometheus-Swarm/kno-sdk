from pathlib import Path
from src.kno_sdk import clone_and_index, EmbeddingMethod, agent_query, search, load_index

# from kno_sdk import clone_and_index, search, EmbeddingMethod, agent_query
import os
from dotenv import load_dotenv

load_dotenv()
# Forking
index = clone_and_index("https://github.com/Prometheus-Swarm/feature-builder", branch="main", embedding=EmbeddingMethod.JINAAI, cloned_repo_base_dir="repos",should_push_to_repo=False)
# print(x)

repo_url = "https://github.com/Prometheus-Swarm/feature-builder"
branch = "main"
system_prompt = f"""
            You are a senior code-analysis agent working on the repository below.

            Your job is to systematically gather information and then summarize your findings.
        """
        
prompt = """
        Before making any changes, can you summarize the architecture and key components of this GitHub repo as you understand it from the current context? 
        Please include the main technologies used, key folders/files, and the primary functionality implemented by reading all the important files.
        If you are missing any crucial files or information, mention that too.
    """
    
format = """f
    ```json
    {{
    "name": "example-project",
    "description": "A cross-platform desktop application for note-taking and task management.",
    "repository_url": "https://github.com/username/example-project",
    
    "primary_language": "C++",
    "languages_used": [
        {{"language": "C++", "percentage": 85.0}},
        {{"language": "QML", "percentage": 10.0}},
        {{"language": "Shell", "percentage": 5.0}}
    ],

    "frameworks_used": [
        {{"name": "Qt", "version": "6.5"}},
        {{"name": "Boost", "version": "1.81"}}
    ],

    "build_tools_used": [
        {{"name": "CMake", "version": "3.27"}},
        {{"name": "Make"}}
    ],

    "test_frameworks_used": [
        {{"name": "Catch2", "version": "3.3"}}
    ],

    "linters_used": [
        {{"name": "clang-tidy"}},
        {{"name": "cppcheck"}}
    ],

    "ci_cd_tools": ["GitHub Actions"],
    "ci_cd_config_files": [".github/workflows/build.yml"],

    "packaging_method": "CMake + CPack",
    "packaging_output_formats": [".tar.gz", ".deb"],

    "deployment_type": "desktop",
    "deployment_platforms": ["Linux", "Windows", "macOS"],

    "application_type": "Desktop",
    "core_features": [
        "Note editing and formatting",
        "Task tagging and reminders",
        "Sync with local filesystem"
    ],

    "authentication_used": false,

    "data_storage_type": "Local",
    "data_storage_format": "SQLite database",
    "data_storage_models": 7,

    "external_dependencies": [
        {{"name": "sqlite", "version": "3.39"}},
        {{"name": "zlib", "version": "1.2.13"}}
    ]
    }}
"""

print("index path", index.path)
# index = load_index(Path("repos/node-express-realworld-example-app"))
resp = agent_query(
    repo_index=index,
    llm_system_prompt=system_prompt,
    prompt=prompt,
    MODEL_API_KEY=os.environ.get("ANTHROPIC_API_KEY"),
    output_format=format,
    embedding=EmbeddingMethod.JINAAI,
)
print(resp)

# y = search("https://github.com/SyedGhazanferAnwar/NestJs-MovieApp", branch="master", embedding=EmbeddingMethod.SBERT, cloned_repo_base_dir="repos",query="NEST")
# print(y)

# Search coming empty everything else ready


# Models to use:
# jinaai/jina-embeddings-v2-base-code
# microsoft/graphcodebert-base



# ISSUE remaining, the chunking might be very bad as the search file always returns empty / bad output
# ISSUE the prompt after the search is very bad, fix that to be better