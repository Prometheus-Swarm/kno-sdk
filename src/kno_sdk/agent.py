import logging
import pathlib
import os
import time
import json
import re
import fnmatch


from pathlib import Path
from git import Repo
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from typing import Dict, List, Tuple, Optional, Any, TypedDict
from enum import Enum

from dataclasses import dataclass
from langchain_anthropic import ChatAnthropic
from langchain.chat_models.base import BaseChatModel
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END

from .embedding import RepoIndex, search

logger = logging.getLogger(__name__)
TOKEN_LIMIT = 16_000  # per-chunk token cap
MAX_ITERATIONS = 30


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# ──────────────────────────── AGENT FACTORY ──────────────────────────
@dataclass
class AgentConfig:
    repo_path: str
    branch: str = "main"
    llm_provider: str = "anthropic"
    model_name: str = "claude-3-5-haiku-latest"
    temperature: float = 0.0
    embedding_function: str = "SBERTEmbedding"
    max_tokens: int = 4096
    cloned_repo_base_dir: str = str(Path.cwd())


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


# ───────────────────────────── TOOLS ────────────────────────────────


def build_tools(index: RepoIndex, llm: LLMProviderBase, cfg: AgentConfig) -> List[Tool]:
    """Return lightweight LangChain `Tool`s that close over *index*."""

    def search_code(query: str, k: int = 8) -> str:
        if not query or query.strip() == "":
            # Return directory structure instead of empty search
            return (
                f"Please provide a search query. Repository structure:\n{index.digest}"
            )
        # If user gives a glob-style file pattern like "*.py"
        if any(char in query for char in ["*", "?", "[", "]"]):
            matching_files = [f for f in index.digest['files'] if fnmatch.fnmatch(f, query)]
            if not matching_files:
                return f"No files match the pattern '{query}'"
            return f"Files matching '{query}':\n" + "\n".join(matching_files)

        # 1) retrieve top‑k code snippets
        snippets = [d.page_content for d in index.vector_store.similarity_search(query, k=k)]
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
        Tool(
            name="search_code",
            func=search_code,
            description="Semantic code search in the repo, Input is a CODE SNIPPET",
        ),
        Tool(
            name="read_file",
            func=read_file,
            description="Read a particular file content in the repo, Input is a file path",
        ),
    ]


# ────────────────────── LANGGRAPH STATE AND NODES ───────────────────────


class AgentState(TypedDict):
    input: str
    repo_info: Dict
    messages: List[Dict[str, Any]]
    intermediate_steps: List[tuple]
    iterations: int


def create_agent_graph(
    tools: List[Tool],
    llm: LLMProviderBase,
    system_message: str,
    output_format: str | None = None,
):
    """Create a LangGraph agent with the provided tools and LLM."""

    # Function to formulate prompt with history and tool results
    def get_prompt_with_history(state: AgentState) -> List[Dict[str, Any]]:
        return (
            [{"role": "system", "content": system_message}]
            + state["messages"]
            + [{"role": "user", "content": state["input"]}]
        )

    # Node 1: Agent thinks about what to do next
    def agent_thinking(state: AgentState) -> AgentState:
        messages = get_prompt_with_history(state)

        # print("-------------------------------------")
        # print("HISTORY",messages)
        # print("-------------------------------------")

        # # Add prompt suffix to guide response format
        # messages.append({"role": "user", "content": prompt_suffix})

        # Get response from LLM
        response = llm.invoke(messages)
        # print("**************************************")

        # print("RESPONSE:", response.content)
        # print("**************************************")

        # Return updated state
        return {
            **state,
            "messages": state["messages"]
            + [{"role": "assistant", "content": response.content}],
        }

    def format_output(state: AgentState) -> AgentState:
        messages = get_prompt_with_history(state)
        output = ""
        if output_format:
            # # Add prompt suffix to guide response format
            messages.append(
                {
                    "role": "user",
                    "content": "FORMAT THE OUTPUT EXACTLY IN THE BELOW FORMAT, with any data not available make it empty string "
                    + output_format,
                }
            )

            # Get response from LLM
            response = llm.invoke(messages)
            output = response.content
            # print("**************************************")
        else:
            response = state["messages"][-1];
            output = response["content"]
        # Return updated state
        return {
            **state,
            "messages": state["messages"]
            + [{"role": "assistant", "content": f"#Final-Answer: {output}"}],
        }

    # Node 2: Parse action and execute tool if needed
    def execute_tools(state: AgentState) -> AgentState:
        """
        • Parse the assistant's most‑recent message for a tool call.
        • Accept either fenced‑JSON *or* the natural‑language pattern
          "I'll use the <tool> tool with input: …".
        • Execute the tool, append the observation, and advance the loop.
        • If no valid tool call is detected, inject a system nudge so the
          agent retries instead of entering an endless loop.
        """
        last_message = state["messages"][-1]["content"] if state["messages"] else ""

        # ---------- exit early on final answer ----------
        if "#Final-Answer:" in last_message:
            return {**state, "iterations": state["iterations"] + 1}

        # ---------- 1. fenced‑JSON tool call ------------
        tool_call = None
        m = re.search(
            r"(?:```json|json)\s*(\{.*?\})\s*(?:```)?", last_message, re.DOTALL
        )
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
                        "role": "assistant",
                        "content": (
                            "I couldn't recognise a tool call. Use `read_file` or `search_code` \n "
                            "Reply either with valid\n"
                            '```json\n{ "action": "tool_name", "action_input": "tool_input" }\n```\n'
                            "or finish with **#Final-Answer:**."
                        ),
                    }
                ],
                "iterations": state["iterations"] + 1,
            }

        tool_name = tool_call.get("action")
        tool_input = tool_call.get("action_input")

        # ───── 4. ***Duplicate‑call guard*** – skip if same as last one ─────
        if state["intermediate_steps"]:
            prev_action, _ = state["intermediate_steps"][-1]
            if (
                isinstance(prev_action, dict)
                and prev_action.get("name") == tool_name
                and prev_action.get("arguments") == tool_input
            ):
                warn_msg = (
                    f"You already ran `{tool_name}` with that exact input. "
                    "I'll skip that and you can choose another action or finish with **#Final-Answer:**."
                )
                # Append a system message (not a new tool step), and do not bump iterations
                return {
                    **state,
                    "messages": state["messages"]
                    + [{"role": "assistant", "content": warn_msg}],
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
                obs_msg = {"role": "user", "content": f"Observation: {result}"}

                return {
                    **state,
                    "messages": state["messages"] + [obs_msg],
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
        if "#Final-Answer:" in last_message:
            return "format_output"

        if state["iterations"] >= MAX_ITERATIONS:
            # force the model to wrap up instead of ending the graph
            state["messages"].append(
                {
                    "role": "user",
                    "content": "You have reached the step limit. Respond now with your output as '#Final-Answer:'.",
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
    workflow.add_node("format_output", format_output)

    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "format_output": "format_output"},
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("format_output", END)

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
        self,
        cfg: AgentConfig,
        index: RepoIndex,
        system_prompt: str = "",
        output_format: str | None = None,
    ):

        llm = self._get_llm(cfg)
        tools = build_tools(index, llm, cfg)
        agent_graph = create_agent_graph(tools, llm, system_prompt, output_format)

        # Create a wrapper that mimics the AgentExecutor.run method
        class AgentGraphRunner:
            def __init__(self, graph):
                self.graph = graph

            def run(self, input_str: str):
                state = {
                    "input": input_str,
                    "repo_info": {
                        "url": cfg.repo_path,
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
                    if (
                        "#Final-Answer:" in last
                        or state["iterations"] >= MAX_ITERATIONS
                    ):
                        break

                match = re.search(r"#Final-Answer:(.*)", last, re.DOTALL)
                return match.group(1).strip() if match else last

        return AgentGraphRunner(agent_graph)


def agent_query(
    repo_index: RepoIndex,
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC,
    llm_model: str = "claude-3-5-haiku-latest",
    llm_temperature: float = 0.0,
    llm_max_tokens: int = 4096,
    llm_system_prompt: str = "",
    prompt: str = "",
    MODEL_API_KEY: str = "",
    output_format: str | None = None,
    embedding: str = "SBERTEmbedding",
):
    if LLMProvider.ANTHROPIC:
        os.environ["ANTHROPIC_API_KEY"] = MODEL_API_KEY
    elif LLMProvider.OPENAI:
        os.environ["OPENAI_API_KEY"] = MODEL_API_KEY
    cfg = AgentConfig(
        repo_path=str(repo_index.path),  # Use full path to local repository
        branch="main",  # Default branch
        llm_provider=llm_provider.value,
        model_name=llm_model,
        embedding_function=embedding,  # Default to SBERT
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
        cloned_repo_base_dir=str(repo_index.path.parent),  # Use parent directory of index
    )
    
    prompt_suffix = """
    You must strictly follow these rules when responding:

    🚨 RULES FOR RESPONSES:

    1. You may **only do ONE of the following per message**:
    - Call a single tool.
    - Respond with a final answer.

    2. If you need more information:
    - Use *only* a single tool call.
    - Your response must be a pure JSON block (no commentary, no Markdown outside the code block).
    - You have 2 tools available: i.e. Search code: `search_code` and Read file: `read_file`.
    - Format it like one of the following:

    ```json
    {
        "action": "search_code",
        "action_input": "<code snippet or query>"
    }
    ```

    OR

    ```json
    {
    "action": "read_file",
    "action_input": "<file_path>"
    }
    ```

    3. If you have all necessary information:
    If you have all necessary information, reply only with:

    #Final-Answer: <your comprehensive answer or solution>
    
    ❌ NEVER mix tool calls and final answers.

    ❌ NEVER include extra explanation or commentary when using a tool.

    ✅ Continue calling tools (one per message) until you are completely ready to give a #Final-Answer.

    Stay disciplined. No tool chaining, no partial answers.
    """
    system_message = (
        f"{llm_system_prompt.strip()}\n\n"
        f"Repository digest:\n{repo_index.digest}\n\n"
        f"{prompt_suffix}"
    )
    agent = AgentFactory().create_agent(
        cfg,
        index=repo_index,
        system_prompt=system_message,
        output_format=output_format,
    )
    result = agent.run(prompt)
    return result
