from src.kno_sdk import clone_and_index, EmbeddingMethod, agent_query, search

# from kno_sdk import clone_and_index, search, EmbeddingMethod, agent_query
import os
from dotenv import load_dotenv

load_dotenv()
# Forking
# x = clone_and_index("https://github.com/gothinkster/node-express-realworld-example-app", branch="master", embedding=EmbeddingMethod.SBERT, cloned_repo_base_dir="repos",should_push_to_repo=False)
# print(x)

repo_url = "https://github.com/gothinkster/node-express-realworld-example-app"
branch = "master"
system_prompt = f"""
            You are a senior code-analysis agent working on the repository below.

            Your job is to systematically gather information and then summarize your findings.

            IMPORTANT RULES:

            * If you need more information, respond ONLY by calling a tool (read_file, search_code).

            * If you have enough information, respond ONLY with
            
            ```
            #Final-Answer: <your comprehensive answer here>
            ```
            
            * "NEVER mix a tool call and a #Final-Answer: in the same message."

            * NEVER include commentary when making a tool call — just the JSON block.

            * Continue gathering information until you are certain you can write a complete #Final-Answer:.

            * Always stay disciplined: TOOL CALL OR FINAL ANSWER — NEVER BOTH TOGETHER.
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