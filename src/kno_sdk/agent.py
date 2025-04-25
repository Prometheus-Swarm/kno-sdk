import logging
import pathlib
import os
import time
import json
import re


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

from .embedding import RepoIndex, clone_and_index, EmbeddingMethod, search

logger = logging.getLogger(__name__)
TOKEN_LIMIT = 16_000  # per-chunk token cap
MAX_ITERATIONS = 30



class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# ──────────────────────────── AGENT FACTORY ──────────────────────────
@dataclass
class AgentConfig:
    repo_url: str
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
        # 1) retrieve top‑k code snippets
        snippets = search(repo_url=cfg.repo_url, branch=cfg.branch, embedding=EmbeddingMethod.SBERT, cloned_repo_base_dir=cfg.cloned_repo_base_dir,query=query, k=k)
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
        Tool(name="search_code", func=search_code, description="Semantic code search in the repo"),
        Tool(name="read_file", func=read_file, description="Read a particular file content in the repo"),
    ]


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
        return (
            [{"role": "system", "content": system_message}]
            + state["messages"]
            + [{"role": "user", "content": state["input"]}]
        )

    # Node 1: Agent thinks about what to do next
    def agent_thinking(state: AgentState) -> AgentState:
        messages = get_prompt_with_history(state)


        # # Add prompt suffix to guide response format
        # messages.append({"role": "user", "content": prompt_suffix})

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
        if "Final Answer:" in last_message and not all(k in tool_call for k in ("action", "action_input")):
            return {**state, "iterations": state["iterations"] + 1}

        # ---------- 1. fenced‑JSON tool call ------------
        tool_call = None
        m = re.search(r"(?:```json|json)\s*(\{.*?\})\s*(?:```)?", last_message, re.DOTALL)
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
                    "content": "You have reached the step limit. Respond now with your output as 'Final Answer:'.",
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
        self, cfg: AgentConfig, cloned_repo_base_dir: str = str(Path.cwd()), system_prompt: str = ""
    ):
        index = clone_and_index(
            cfg.repo_url, cfg.branch, cfg.embedding_function, cloned_repo_base_dir, False
        )
        llm = self._get_llm(cfg)
        tools = build_tools(index, llm, cfg)
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
    

def agent_query(
    repo_url: str,
    branch: str = "main",
    embedding: EmbeddingMethod = EmbeddingMethod.SBERT,
    cloned_repo_base_dir: str = str(Path.cwd()),
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC,
    llm_model: str = "claude-3-5-haiku-latest",
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
        cloned_repo_base_dir=cloned_repo_base_dir
    )
    prompt_suffix = """
            When you need to call a tool you MUST respond with *only* fenced JSON,
            like this – no commentary, no Markdown other than the three back-ticks:
                    
            1. Use a tool to gather more information by formatting your response as:
            ```json
            { "action": "search_code",
            "action_input": "<QUERY>"}
            
            OR
            
            ```json
            { "action": "read_file",
            "action_input": "<FILE_PATH>" } }   
             
            When you are done, reply with:
            
            ```
            Final Answer: Your comprehensive analysis or solution here.
            ```
            IMPORTANT: Do not respond with tool call and Final Answer in one response.
        """
    system_message = llm_system_prompt.strip() + "\n\n" + prompt_suffix

    agent = AgentFactory().create_agent(
        cfg, cloned_repo_base_dir=cloned_repo_base_dir, system_prompt=system_message
    )
    result = agent.run(prompt)
    return result
