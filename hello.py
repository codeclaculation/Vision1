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

endpoint = ""
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














import json
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from openai import OpenAI
import trino
import logging
from datetime import datetime


# =====================================================
# 1️⃣ Enhanced LLM Client (Modular Inference)
# =====================================================

class LLMClient:
    def __init__(self, endpoint: str, api_key: str, model: str = None):
        self.client = OpenAI(api_key=api_key, base_url=endpoint)
        self.model = model or self.get_default_model()

    def get_default_model(self):
        try:
            models = self.client.models.list()
            return models.data[0].id
        except Exception as e:
            logging.error(f"Error fetching default model: {e}")
            raise RuntimeError("Failed to fetch default LLM model")

    def ask(self, prompt: str, system_msg: Optional[str] = None) -> str:
        """Unified interface for all inference calls with error handling."""
        try:
            messages = [{"role": "system", "content": system_msg or "You are an assistant."}]
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                top_p=0.9,
                max_tokens=1200,
                stop=["<|eot_id|>"]
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"LLM inference error: {e}")
            raise RuntimeError("Failed to process LLM request")


# =====================================================
# 2️⃣ Enhanced State Definition
# =====================================================

class GraphState(BaseModel):
    user_query: str
    data_product_id: Optional[str] = None
    schema_info: Optional[str] = None
    sql_query: Optional[str] = None
    validation_result: Optional[str] = None
    execution_result: Optional[Any] = None
    insights: Optional[str] = None
    next_step: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =====================================================
# 3️⃣ Instantiate LLM Client and Trino Connection
# =====================================================

endpoint = " "
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
# 4️⃣ Graph Nodes (Improved with Error Handling)
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
    try:
        response = llm_client.ask(prompt)
        result = json.loads(response)
        state.data_product_id = result.get("data_product_id")
        state.logs.append(f"Identified data product: {state.data_product_id}")
        state.next_step = "metadata_retriever" if result.get("is_appropriate") else END
    except Exception as e:
        logging.error(f"Intent parsing error: {e}")
        state.logs.append("Failed to parse intent")
        state.next_step = END
    return state


def metadata_retriever(state: GraphState) -> GraphState:
    """Fetch dynamic metadata/schema for the identified data product"""
    try:
        cursor = trino_conn.cursor()
        cursor.execute(f"DESCRIBE {state.data_product_id}")
        columns = cursor.fetchall()
        schema_info = ", ".join([f"{col[0]} ({col[1]})" for col in columns])
        state.schema_info = f"Schema for dataset `{state.data_product_id}`: {schema_info}"
        state.logs.append(f"Fetched schema: {state.schema_info}")
        state.next_step = "sql_generator"
    except Exception as e:
        logging.error(f"Metadata retrieval error: {e}")
        state.logs.append("Failed to retrieve metadata")
        state.next_step = END
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
    try:
        sql_query = llm_client.ask(prompt)
        state.sql_query = sql_query.strip()
        state.logs.append(f"Generated SQL: {state.sql_query}")
        state.next_step = "sql_validator"
    except Exception as e:
        logging.error(f"SQL generation error: {e}")
        state.logs.append("Failed to generate SQL")
        state.next_step = END
    return state


# Other nodes (sql_validator, query_executor, etc.) follow similar patterns with enhanced error handling and logging.

# =====================================================
# 5️⃣ Suggestions for Further Improvements
# =====================================================

1. **Interactive Front-End:**
   - Use tools like `Streamlit` or `Gradio` to create an interactive UI where users can input queries and see real-time results with charts.

2. **Parallelism:**
   - Use libraries like Ray or Prefect to parallelize metadata retrieval and validation.

3. **Caching:**
   - Add caching for frequent queries and schemas to reduce latency.

4. **Visualization:**
   - Format the final output as JSON for integration with Plotly or Chart.js.

Let me know if you'd like me to expand on specific nodes or provide additional code for the front end!
