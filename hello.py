"""
LangGraph orchestrator agent with 5 parallel LLM tools.
All LLM inference calls use your LLMClient.ask() function (blocking),
wrapped into asyncio via run_in_executor for parallelism.

Outputs: FiveWayAnalysis Pydantic model containing the five adjacent variables.
"""

import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# ---------- Your LLMClient (uses OpenAI-compatible / internal endpoint) ----------
from openai import OpenAI  # uses OpenAI-compatible client per your environment

class LLMClient:
    def __init__(self, endpoint: str, api_key: str, model: Optional[str] = None):
        self.client = OpenAI(api_key=api_key, base_url=endpoint)
        self.model = model or self.get_default_model()

    def get_default_model(self) -> str:
        models = self.client.models.list()
        return models.data[0].id

    def ask(self, prompt: str, temperature: float = 0.0, top_p: float = 0.9, max_tokens: int = 800, stop: Optional[List[str]] = None) -> str:
        """Blocking call to the internal LLM endpoint"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an enterprise Text→SQL analysis assistant. Respond in JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or ["<|eot_id|>"]
        )
        choice = resp.choices[0]
        # robust extraction for different response shapes
        if hasattr(choice, "message") and getattr(choice.message, "content", None):
            return choice.message.content
        if isinstance(choice, dict) and "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
        if "text" in choice:
            return choice["text"]
        return json.dumps(choice)

# ---------- Pydantic models for outputs ----------
class Relevancy(BaseModel):
    is_relevant: bool
    score: float
    explanation: str

class Safety(BaseModel):
    is_safe: bool
    flags: List[str]
    explanation: str

class CategorizedRules(BaseModel):
    date_time_rules: List[int] = Field(default_factory=list)
    sql_rules: List[int] = Field(default_factory=list)
    examples: List[int] = Field(default_factory=list)

class FiveWayAnalysis(BaseModel):
    relevancy: Relevancy
    safety: Safety
    business_rules_applicable: List[int]
    categorized_rules: CategorizedRules
    sql_rule_indices: List[int]

# ---------- Prompts for every tool (strict JSON output requested) ----------
RELEVANCY_PROMPT = """
Return JSON exactly:
{
  "is_relevant": true|false,
  "score": 0.0-1.0,
  "explanation": "one-sentence reason"
}
User query: {user_query}
Product rules: {product_rules}
"""

SAFETY_PROMPT = """
Return JSON exactly:
{
  "is_safe": true|false,
  "flags": ["PII","Destructive_SQL", ...],
  "explanation": "one-sentence reason"
}
User query: {user_query}
Product rules: {product_rules}
"""

BUSINESS_RULES_APPLICABLE_PROMPT = """
Return JSON exactly:
{
  "applicable_rule_indices": [0,2,5]
}
User query: {user_query}
Product rules (indexed from 0): {product_rules}
"""

CATEGORIZED_RULES_PROMPT = """
Return JSON exactly:
{
  "date_time_rules": [indices],
  "sql_rules": [indices],
  "examples": [indices]
}
Product rules: {product_rules}
User query: {user_query}
(If a rule fits multiple categories, include its index in each applicable list.)
"""

SQL_RULES_PROMPT = """
Return JSON exactly:
{
  "sql_rule_indices": [indices]
}
Product rules: {product_rules}
User query: {user_query}
"""

# ---------- Executor for wrapping blocking LLMClient.ask into async ----------
_executor = ThreadPoolExecutor(max_workers=6)

async def async_ask(llm: LLMClient, prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: llm.ask(prompt))

# ---------- Utility: parse JSON safely from LLM text ----------
def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempts to parse JSON from text. First try json.loads directly.
    If fails, try to find the first {...} block and parse that.
    """
    if not text or not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
    return None

# ---------- The five async tool functions (each calls llm_client.ask via async_ask) ----------
async def relevance_tool(user_query: str, product_rules: List[str], llm_client: LLMClient) -> Relevancy:
    prompt = RELEVANCY_PROMPT.format(user_query=user_query, product_rules=json.dumps(product_rules))
    raw = await async_ask(llm_client, prompt)
    parsed = extract_json_from_text(raw)
    if not parsed:
        # fallback conservative
        return Relevancy(is_relevant=True, score=0.5, explanation="Could not parse LLM output; defaulted to relevant.")
    return Relevancy(
        is_relevant=bool(parsed.get("is_relevant", True)),
        score=float(parsed.get("score", 0.5)),
        explanation=str(parsed.get("explanation", ""))
    )

async def safety_tool(user_query: str, product_rules: List[str], llm_client: LLMClient) -> Safety:
    prompt = SAFETY_PROMPT.format(user_query=user_query, product_rules=json.dumps(product_rules))
    raw = await async_ask(llm_client, prompt)
    parsed = extract_json_from_text(raw)
    if not parsed:
        return Safety(is_safe=True, flags=[], explanation="Could not parse LLM safety output; default safe.")
    return Safety(
        is_safe=bool(parsed.get("is_safe", True)),
        flags=list(parsed.get("flags", []) or []),
        explanation=str(parsed.get("explanation", ""))
    )

async def business_rules_applicable_tool(user_query: str, product_rules: List[str], llm_client: LLMClient) -> List[int]:
    prompt = BUSINESS_RULES_APPLICABLE_PROMPT.format(user_query=user_query, product_rules=json.dumps(product_rules))
    raw = await async_ask(llm_client, prompt)
    parsed = extract_json_from_text(raw)
    if parsed and "applicable_rule_indices" in parsed:
        return [int(x) for x in parsed.get("applicable_rule_indices", [])]
    return []

async def categorize_rules_tool(user_query: str, product_rules: List[str], llm_client: LLMClient) -> CategorizedRules:
    prompt = CATEGORIZED_RULES_PROMPT.format(user_query=user_query, product_rules=json.dumps(product_rules))
    raw = await async_ask(llm_client, prompt)
    parsed = extract_json_from_text(raw)
    if not parsed:
        return CategorizedRules()
    return CategorizedRules(
        date_time_rules=[int(x) for x in parsed.get("date_time_rules", [])],
        sql_rules=[int(x) for x in parsed.get("sql_rules", [])],
        examples=[int(x) for x in parsed.get("examples", [])]
    )

async def sql_rules_tool(user_query: str, product_rules: List[str], llm_client: LLMClient) -> List[int]:
    prompt = SQL_RULES_PROMPT.format(user_query=user_query, product_rules=json.dumps(product_rules))
    raw = await async_ask(llm_client, prompt)
    parsed = extract_json_from_text(raw)
    if parsed and "sql_rule_indices" in parsed:
        return [int(x) for x in parsed.get("sql_rule_indices", [])]
    return []

# ---------- Orchestrator agent (LangGraph node) ----------
async def orchestrator_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph agent node that:
    - accepts state with keys: 'user_query', 'product_rules' (list)
    - runs the five async LLM tools in parallel
    - attaches the FiveWayAnalysis into state['analysis'] and returns state
    """
    user_query = state.get("user_query", "")
    product_rules = state.get("product_rules", []) or []

    # instantiate LLM client from environment (or already passed in state)
    endpoint = state.get("llm_endpoint") or os.environ.get("LOCAL_LLM_ENDPOINT")
    api_key = state.get("llm_api_key") or os.environ.get("LOCAL_LLM_API_KEY")
    model = state.get("llm_model") or os.environ.get("LOCAL_LLM_MODEL")
    llm_client = LLMClient(endpoint, api_key, model)

    # Run five LLM-based tools in parallel
    results = await asyncio.gather(
        relevance_tool(user_query, product_rules, llm_client),
        safety_tool(user_query, product_rules, llm_client),
        business_rules_applicable_tool(user_query, product_rules, llm_client),
        categorize_rules_tool(user_query, product_rules, llm_client),
        sql_rules_tool(user_query, product_rules, llm_client),
        return_exceptions=False
    )

    relevancy_obj, safety_obj, applicable_indices, categorized_obj, sql_indices = results

    # Build Pydantic model with adjacent outputs
    analysis = FiveWayAnalysis(
        relevancy=relevancy_obj,
        safety=safety_obj,
        business_rules_applicable=applicable_indices,
        categorized_rules=categorized_obj,
        sql_rule_indices=sql_indices
    )

    # place results into state and return
    state["analysis"] = analysis.dict()
    return state

# ---------- LangGraph wiring ----------
# Define a simple typed dict state for the graph (LangGraph accepts plain dicts too)
GraphState = TypedDict("GraphState", {"user_query": str, "product_rules": list, "analysis": dict})

graph = StateGraph(dict)
graph.add_node("orchestrator_agent", orchestrator_agent)
graph.add_edge(START, "orchestrator_agent")
graph.add_edge("orchestrator_agent", END)

compiled = graph.compile()

# ---------- Runner: invoke the compiled graph (async) ----------
async def run_orchestration(user_query: str, product_rules: List[str]):
    initial_state = {"user_query": user_query, "product_rules": product_rules}
    # If compiled supports async invocation use ainvoke/stream; fallback to invoke in event loop
    # Try compiled.ainvoke if available, else call orchestrator_agent directly.
    try:
        # Prefer graph async invocation
        if hasattr(compiled, "ainvoke"):
            final_state = await compiled.ainvoke(initial_state)
        else:
            # If compile returns callable that is sync, call node directly (it will run our coroutine)
            # compiled.invoke may run coroutine internally; attempt to call via run_in_executor
            loop = asyncio.get_event_loop()
            final_state = await loop.run_in_executor(None, lambda: compiled.invoke(initial_state))
    except Exception as e:
        # Last-resort: call orchestrator_agent directly
        final_state = await orchestrator_agent(initial_state)

    return final_state

# ---------- Example usage ----------
if __name__ == "__main__":
    # Environment variables or override here
    os.environ["LOCAL_LLM_ENDPOINT"] = "http://njp@1gpu34.sdi.corp.bankofamerica.com:8097/v1"
    os.environ["LOCAL_LLM_API_KEY"] = "sk-password123"
    # optional: os.environ["LOCAL_LLM_MODEL"] = "your-model-id"

    product_rules_example = [
        "Always include LIMIT 1000 in preview queries.",
        "Do not expose customer_id or PII fields.",
        "Default aggregation: SUM(sales_amount) when user asks for 'total sales'.",
        "Date filters must be applied for 'last month' or 'last quarter' requests.",
        "No cross-catalog joins are allowed.",
        "Use date_trunc('month', date) for monthly rollups.",
        "Avoid wide scans by including region or product filter when available."
    ]

    user_q = "Show total sales by region for the last quarter including customer_id."

    final = asyncio.run(run_orchestration(user_q, product_rules_example))

    print("\n--- Five adjacent outputs (analysis) ---\n")
    print(json.dumps(final.get("analysis", {}), indent=2))