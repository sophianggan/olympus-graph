"""
Olympus Graph – LangGraph Self-Correcting Agent Workflow

Workflow (5 nodes):
  1. Parser     → Decides: Historical Data (Cypher) or Future Prediction (GNN)
  2. Generator  → Generates the Cypher query or GNN parameters
  3. Executor   → Runs the query/model
  4. Reflector  → Error handling + self-correction loop
  5. Answer     → Final natural-language synthesis

Uses LangGraph (NOT LangChain chains).
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict, Annotated

from loguru import logger

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.agent.tools import (
    graph_query_tool,
    model_predict_tool,
    GRAPH_SCHEMA_DESCRIPTION,
)


# ══════════════════════════════════════════════════════
# State Schema
# ══════════════════════════════════════════════════════

class AgentState(TypedDict):
    """State that flows through the LangGraph workflow."""
    user_query: str
    query_type: str  # "historical" | "prediction" | "unknown"
    generated_code: str  # Cypher query or tool parameters
    execution_result: dict[str, Any]
    error_message: str
    retry_count: int
    final_answer: str
    messages: list[dict]  # Conversation history


# ══════════════════════════════════════════════════════
# LLM Setup
# ══════════════════════════════════════════════════════

def get_llm() -> ChatOpenAI:
    """Initialize the LLM for the agent."""
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-your-key-here"):
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to `.env` before using the LangGraph agent."
        )
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.1,
    )


# ══════════════════════════════════════════════════════
# Node 1: Parser
# ══════════════════════════════════════════════════════

PARSER_SYSTEM_PROMPT = """You are an Olympic data analyst. Given a user query, classify it as:
- "historical": Questions about past Olympic data (e.g., "Who won Gold in 100m in 2020?")
- "prediction": Questions about future Olympic predictions (e.g., "Who will win the 100m in 2028?")
- "unknown": Cannot be answered with available tools

Respond with ONLY one word: "historical", "prediction", or "unknown"."""


def parser_node(state: AgentState) -> dict:
    """Classify the user query as historical or prediction."""
    llm = get_llm()
    query = state["user_query"]

    response = llm.invoke([
        SystemMessage(content=PARSER_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ])

    query_type = response.content.strip().lower().strip('"\'')

    # Fallback heuristics
    future_keywords = ["will", "predict", "2025", "2026", "2027", "2028", "2032", "future", "next"]
    past_keywords = ["won", "did", "was", "were", "who won", "history", "past"]

    if query_type not in ("historical", "prediction", "unknown"):
        query_lower = query.lower()
        if any(kw in query_lower for kw in future_keywords):
            query_type = "prediction"
        elif any(kw in query_lower for kw in past_keywords):
            query_type = "historical"
        else:
            query_type = "historical"  # default

    logger.info(f"Parser classified query as: {query_type}")
    return {"query_type": query_type}


# ══════════════════════════════════════════════════════
# Node 2: Generator
# ══════════════════════════════════════════════════════

CYPHER_GENERATOR_PROMPT = f"""You are a Neo4j Cypher expert for the Olympic Knowledge Graph.

{GRAPH_SCHEMA_DESCRIPTION}

Given a user question, generate ONLY a valid Cypher query. No explanation, just the query.
Return results limited to 25 rows max.
Always use RETURN with meaningful aliases."""

PREDICT_GENERATOR_PROMPT = """You are an Olympic prediction assistant.
Given a user question about future Olympic predictions, extract:
1. event_name: The event (e.g., "Men's 100 metres")
2. target_year: The year (default 2028)
3. top_k: How many predictions (default 3)

Respond with ONLY a JSON object like:
{"event_name": "...", "target_year": 2028, "top_k": 3}"""


def generator_node(state: AgentState) -> dict:
    """Generate Cypher query or prediction parameters."""
    llm = get_llm()
    query = state["user_query"]
    query_type = state["query_type"]
    error = state.get("error_message", "")

    if query_type == "historical":
        messages = [
            SystemMessage(content=CYPHER_GENERATOR_PROMPT),
            HumanMessage(content=query),
        ]
        if error:
            messages.append(HumanMessage(
                content=f"The previous query failed with error: {error}\n"
                        f"Please fix the Cypher query."
            ))

        response = llm.invoke(messages)
        generated = response.content.strip()

        # Clean up: remove markdown code fences if present
        if generated.startswith("```"):
            lines = generated.split("\n")
            generated = "\n".join(
                l for l in lines if not l.startswith("```")
            ).strip()

        logger.info(f"Generated Cypher: {generated[:100]}...")
        return {"generated_code": generated}

    elif query_type == "prediction":
        messages = [
            SystemMessage(content=PREDICT_GENERATOR_PROMPT),
            HumanMessage(content=query),
        ]
        if error:
            messages.append(HumanMessage(
                content=f"Previous prediction failed: {error}\n"
                        f"Please adjust the parameters."
            ))

        response = llm.invoke(messages)
        generated = response.content.strip()

        # Clean up JSON
        if generated.startswith("```"):
            lines = generated.split("\n")
            generated = "\n".join(
                l for l in lines if not l.startswith("```")
            ).strip()

        logger.info(f"Generated prediction params: {generated}")
        return {"generated_code": generated}

    else:
        return {"generated_code": ""}


# ══════════════════════════════════════════════════════
# Node 3: Executor
# ══════════════════════════════════════════════════════

def executor_node(state: AgentState) -> dict:
    """Execute the generated query or prediction."""
    query_type = state["query_type"]
    code = state["generated_code"]

    if query_type == "historical":
        result = graph_query_tool(code)
        return {"execution_result": result, "error_message": "" if result["success"] else result["data"]}

    elif query_type == "prediction":
        try:
            params = json.loads(code)
            result = model_predict_tool(
                event_name=params.get("event_name", ""),
                target_year=params.get("target_year", 2028),
                top_k=params.get("top_k", 3),
            )
            return {
                "execution_result": result,
                "error_message": "" if result["success"] else result.get("error", "Unknown error"),
            }
        except json.JSONDecodeError as e:
            return {
                "execution_result": {"success": False},
                "error_message": f"Invalid JSON parameters: {e}",
            }

    return {
        "execution_result": {"success": False, "data": "Unknown query type"},
        "error_message": "Query type not recognized",
    }


# ══════════════════════════════════════════════════════
# Node 4: Reflector (Self-Correction)
# ══════════════════════════════════════════════════════

MAX_RETRIES = 3


def reflector_node(state: AgentState) -> dict:
    """
    Check execution result. If failed or empty, increment retry
    and prepare error message for the Generator to fix.
    """
    result = state["execution_result"]
    retry = state.get("retry_count", 0)

    if result.get("success"):
        # Check for empty results
        data = result.get("data", [])
        if isinstance(data, list) and len(data) == 0:
            if retry < MAX_RETRIES:
                logger.warning("Empty results, asking Generator to rewrite")
                return {
                    "error_message": "The query returned empty results. Try broadening the search or using different filters.",
                    "retry_count": retry + 1,
                }
        # Success — clear error
        return {"error_message": "", "retry_count": retry}

    # Execution failed
    if retry < MAX_RETRIES:
        logger.warning(f"Execution failed (retry {retry + 1}/{MAX_RETRIES}): {state.get('error_message', '')}")
        return {"retry_count": retry + 1}
    else:
        logger.error(f"Max retries ({MAX_RETRIES}) reached")
        return {"retry_count": retry}


# ══════════════════════════════════════════════════════
# Node 5: Answer Synthesizer
# ══════════════════════════════════════════════════════

ANSWER_PROMPT = """You are an Olympic expert AI assistant. Given the query results, 
provide a clear, informative answer with:
1. Direct answer to the question
2. Supporting data/statistics
3. For predictions: explain WHY certain athletes are favored (momentum, country hosting advantage, etc.)

Be concise but insightful. Use markdown formatting for tables when showing rankings."""


def answer_node(state: AgentState) -> dict:
    """Synthesize the final natural-language answer."""
    llm = get_llm()
    query = state["user_query"]
    result = state["execution_result"]
    query_type = state["query_type"]

    if not result.get("success") and state.get("retry_count", 0) >= MAX_RETRIES:
        return {
            "final_answer": (
                f"I'm sorry, I couldn't find a reliable answer to your question: "
                f"'{query}'. The error was: {state.get('error_message', 'Unknown')}. "
                f"Please try rephrasing your question."
            )
        }

    context = json.dumps(result, indent=2, default=str)

    messages = [
        SystemMessage(content=ANSWER_PROMPT),
        HumanMessage(content=(
            f"User Question: {query}\n\n"
            f"Query Type: {query_type}\n\n"
            f"Data/Results:\n{context}"
        )),
    ]

    response = llm.invoke(messages)
    answer = response.content.strip()

    logger.info(f"Generated answer ({len(answer)} chars)")
    return {"final_answer": answer}


# ══════════════════════════════════════════════════════
# Routing Logic
# ══════════════════════════════════════════════════════

def should_retry(state: AgentState) -> str:
    """Decide whether to retry (loop back to Generator) or proceed to Answer."""
    result = state.get("execution_result", {})
    retry = state.get("retry_count", 0)
    error = state.get("error_message", "")

    if error and retry < MAX_RETRIES:
        return "generator"  # Loop back for self-correction
    return "answer"


# ══════════════════════════════════════════════════════
# Build the LangGraph Workflow
# ══════════════════════════════════════════════════════

def build_agent_graph() -> StateGraph:
    """
    Build the LangGraph workflow:

    START → Parser → Generator → Executor → Reflector → (retry?) → Answer → END
                         ↑                                  |
                         └──────── (on error) ──────────────┘
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("parser", parser_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reflector", reflector_node)
    workflow.add_node("answer", answer_node)

    # Define edges
    workflow.add_edge(START, "parser")
    workflow.add_edge("parser", "generator")
    workflow.add_edge("generator", "executor")
    workflow.add_edge("executor", "reflector")

    # Conditional: Reflector → retry Generator or → Answer
    workflow.add_conditional_edges(
        "reflector",
        should_retry,
        {
            "generator": "generator",
            "answer": "answer",
        },
    )

    workflow.add_edge("answer", END)

    return workflow.compile()


# ══════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════

_agent = None


def get_agent():
    """Return a singleton compiled agent."""
    global _agent
    if _agent is None:
        _agent = build_agent_graph()
    return _agent


def ask(query: str) -> str:
    """
    Ask the Olympus Graph agent a question.

    Args:
        query: Natural language question about Olympics

    Returns:
        Natural language answer
    """
    try:
        agent = get_agent()
    except ValueError as exc:
        return str(exc)

    initial_state: AgentState = {
        "user_query": query,
        "query_type": "",
        "generated_code": "",
        "execution_result": {},
        "error_message": "",
        "retry_count": 0,
        "final_answer": "",
        "messages": [],
    }

    try:
        result = agent.invoke(initial_state)
        return result["final_answer"]
    except ValueError as exc:
        return str(exc)


# ── CLI Entry Point ──────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "Who won the Men's 100m at the 2020 Olympics?"

    print(f"\n🏛️  Olympus Graph Agent")
    print(f"📝 Query: {question}\n")
    answer = ask(question)
    print(f"💬 Answer:\n{answer}")
