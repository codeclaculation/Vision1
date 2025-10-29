import json
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from openai import OpenAI
import trino

# =====================================================
# 1️⃣ Custom LLM Client (Your Internal Inference Endpoint)
# =====================================================

class LLMClient:
    def __init__(self, endpoint: str, api_key: str, model: str = None):
        self.client = OpenAI(api_key=api_key, base_url=endpoint)
        self.model = model or self.get_default_model()

    def get_default_model(self):
        models = self.client.models.list()
        return models.data[0].id

    def ask(self, prompt: str) -> str:
        """Unified interface for all inference calls"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a SQL and data reasoning assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            top_p=0.9,
            max_tokens=1200,
            stop=["<|eot_id|>"]
        )
        return response.choices[0].message.content


# =====================================================
# 2️⃣ State Definition
# =====================================================

class GraphState(BaseModel):
    user_query: str
    data_product_id: str = None
    schema_info: str = None
    sql_query: str = None
    validation_result: str = None
    execution_result: Any = None
    insights: str = None
    next_step: str = None


# =====================================================
# 3️⃣ Instantiate LLM Client and Trino Connection
# =====================================================

endpoint = "http://njp@1gpu34.sdi.corp.bk.com:8097/v1"
api_key = "your_internal_api_key"
llm_client = LLMClient(endpoint, api_key, model="llama-70b")

trino_conn = trino.dbapi.connect(
    host="your_trino_host",
    port=8080,
    user="data_engineer",
    catalog="hive",
    schema="default",
)

# =====================================================
# 4️⃣ Graph Nodes
# =====================================================

def parse_intent(state: GraphState) -> GraphState:
    """Use LLM to understand user's query and identify data product"""
    prompt = f"""
    You are an AI system that determines which data product should be used for a query.

    User query: {state.user_query}

    Respond strictly in JSON:
    {{
      "data_product_id": "<dataset>",
      "is_appropriate": true/false,
      "reason": "<why>"
    }}
    """
    response = llm_client.ask(prompt)
    try:
        result = json.loads(response)
        state.data_product_id = result.get("data_product_id")
        state.next_step = "metadata_retriever" if result.get("is_appropriate") else END
    except Exception:
        state.next_step = END
    return state


def metadata_retriever(state: GraphState) -> GraphState:
    """Fetch metadata/schema for the identified data product"""
    schema_info = f"Fetched schema for dataset `{state.data_product_id}` with columns: id, amount, date, region"
    state.schema_info = schema_info
    state.next_step = "sql_generator"
    return state


def sql_generator(state: GraphState) -> GraphState:
    """Generate a SQL query based on user intent and schema"""
    prompt = f"""
    You are a SQL query generator for Trino.
    Based on the schema below, generate a valid SQL query for the user question.

    Schema:
    {state.schema_info}

    User question:
    {state.user_query}

    Output only SQL (no explanation).
    """
    sql_query = llm_client.ask(prompt)
    state.sql_query = sql_query.strip()
    state.next_step = "sql_validator"
    return state


def sql_validator(state: GraphState) -> GraphState:
    """Validate SQL syntax and policy compliance"""
    prompt = f"""
    Validate the following SQL for syntax and data governance compliance in Trino.
    SQL:
    {state.sql_query}

    Respond in JSON with:
    {{
      "is_valid": true/false,
      "message": "<validation summary>"
    }}
    """
    response = llm_client.ask(prompt)
    try:
        validation = json.loads(response)
        state.validation_result = validation["message"]
        if validation["is_valid"]:
            state.next_step = "query_executor"
        else:
            state.next_step = END
    except Exception:
        state.validation_result = "Validation failed due to parsing error."
        state.next_step = END
    return state


def query_executor(state: GraphState) -> GraphState:
    """Execute SQL and fetch result preview"""
    try:
        cursor = trino_conn.cursor()
        cursor.execute(state.sql_query)
        rows = cursor.fetchall()
        state.execution_result = rows[:5]
        state.next_step = "insight_summarizer"
    except Exception as e:
        state.execution_result = f"Query execution error: {e}"
        state.next_step = END
    return state


def insight_summarizer(state: GraphState) -> GraphState:
    """Generate summary insights from query results"""
    prompt = f"""
    Summarize the following result set and provide key business insights.
    Results (first 5 rows): {state.execution_result}
    """
    insights = llm_client.ask(prompt)
    state.insights = insights
    state.next_step = "audit_logger"
    return state


def audit_logger(state: GraphState) -> GraphState:
    """Log the transaction details for observability"""
    log_entry = {
        "query": state.user_query,
        "dataset": state.data_product_id,
        "sql": state.sql_query,
        "validation": state.validation_result,
        "insights": state.insights,
    }
    print(f"[AUDIT LOG]: {json.dumps(log_entry, indent=2)}")
    state.next_step = END
    return state


# =====================================================
# 5️⃣ LangGraph Definition and Routing
# =====================================================

graph = StateGraph(GraphState)

graph.add_node("parse_intent", parse_intent)
graph.add_node("metadata_retriever", metadata_retriever)
graph.add_node("sql_generator", sql_generator)
graph.add_node("sql_validator", sql_validator)
graph.add_node("query_executor", query_executor)
graph.add_node("insight_summarizer", insight_summarizer)
graph.add_node("audit_logger", audit_logger)

graph.set_entry_point("parse_intent")

graph.add_edge("parse_intent", "metadata_retriever")
graph.add_edge("metadata_retriever", "sql_generator")
graph.add_edge("sql_generator", "sql_validator")
graph.add_edge("sql_validator", "query_executor")
graph.add_edge("query_executor", "insight_summarizer")
graph.add_edge("insight_summarizer", "audit_logger")
graph.add_edge("audit_logger", END)

app = graph.compile()

# =====================================================
# 6️⃣ Run Inference
# =====================================================

if __name__ == "__main__":
    query = "Show me the total transaction amount per region for the last quarter."
    initial_state = GraphState(user_query=query)
    final_state = app.invoke(initial_state)
    print("\n✅ Final Insights:\n", final_state.insights)
