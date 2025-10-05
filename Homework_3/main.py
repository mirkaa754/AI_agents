import os, operator, sqlite3, json, asyncio, re
from typing import Annotated, Sequence, Dict, List

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from langchain_tavily import TavilySearch
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper as Wikipedia
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "products.db")

def _ensure_db():
    # create directory + db with demo data
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    init_needed = not os.path.exists(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    if init_needed:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS products(
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL
        )""")
        cur.executemany("INSERT INTO products VALUES(?,?,?,?)", [
            (1,"Dell XPS 13","Electronics",1299.99),
            (2,"Apple iPhone 15","Electronics",1099.00),
            (3,"Ergo Desk Chair","Furniture",329.90),
            (4,"Python Tricks (Book)","Books",39.00),
            (5,"DeLonghi Coffee Maker","Appliances",85.50),
        ])
        conn.commit()
    conn.close()


@tool
def sql_query(query: str) -> str:
    """Run read-only SELECT against local data/products.db (table: products)."""
    try:
        if not query.strip().upper().startswith("SELECT"):
            return "Only SELECT queries are allowed."
        _ensure_db()
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        conn.close()
        if not rows: 
            return "No rows."
        lines = [" | ".join(cols)]
        for r in rows: 
            lines.append(" | ".join(str(v) for v in r))
        return "\n".join(lines)
    except Exception as e:
        return f"SQL error opening or querying DB at: {DB_PATH}\nError: {e}"
    
tavily = TavilySearch(max_results=4)
wikipedia = WikipediaQueryRun(api_wrapper=Wikipedia())
wolfram = WolframAlphaQueryRun(api_wrapper=WolframAlphaAPIWrapper())

ALL_TOOLS = {"tavily_search": tavily, "wikipedia_search": wikipedia, "wolfram_math": wolfram, "sqlite": sql_query}

class State(dict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    plan: List[str]
    observations: Dict[str, str]

def planner(state: State):
    last_user = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    if re.match(r"^\s*select\s", last_user, re.I):
        tools = ["sqlite"]
        return {"plan": tools, "messages":[AIMessage(content=f"Plan: {tools}\nReason: Direct SQL detected")]}
    
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"), temperature=0)
    system = (
        "You are a ReAct planner. Choose tools from ['tavily_search','wikipedia_search','wolfram_math','sqlite']."
        "Return JSON: {\"tools\":[],\"rationale\":\"...\"}. Use multiple if helpful. Use 'sqlite' only for DB questions."
    )
    prompt = [{"role":"system","content":system}] + [{"role":"user" if m.type=='human' else "assistant","content":m.content} for m in state["messages"]]
    resp = llm.invoke(prompt)
    text = resp.content or ""
    m = re.search(r"\{.*\}", text, re.S)
    tools = []
    rationale = ""
    if m:
        try:
            obj = json.loads(m.group(0))
            tools = [t for t in obj.get("tools",[]) if t in ALL_TOOLS]
            rationale = obj.get("rationale","")
        except Exception:
            pass
    if not tools: tools = ["tavily_search"]
    return {"plan": tools, "messages":[AIMessage(content=f"Plan: {tools}\nReason: {rationale}")]}

async def _run_one(tool_name: str, question: str) -> str:
    tool = ALL_TOOLS[tool_name]
    try:
        if tool_name == "sqlite":
            # naive mapping: if user didn't write SQL, provide a default helpful query
            if not question.strip().lower().startswith("select"):
                return tool.invoke("SELECT * FROM products WHERE category='Electronics'")
            return tool.invoke(question)
        elif hasattr(tool, "invoke"):
            return tool.invoke(question)
        else:
            return tool.run(question)
    except Exception as e:
        return f"{tool_name} error: {e}"

def fanout(state: State):
    question = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    tools = state.get("plan", [])
    async def gather():
        return await asyncio.gather(*[_run_one(t, question) for t in tools])
    observations = dict(zip(tools, asyncio.run(gather())))
    # summarize
    report = "Parallel tool results:\\n\\n" + "\\n\\n".join([f"### {k}\\n{(str(v)[:500] + '...') if len(str(v))>500 else v}" for k,v in observations.items()])
    return {"observations": observations, "messages":[AIMessage(content=report)]}

def synthesize(state: State):
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"), temperature=0)
    system = (
        "Use the observations to answer. Mention tools you used (Tavily, Wikipedia, Wolfram Alpha, SQLite). "
        "Be concise and accurate."
    )
    msgs = [{"role":"system","content":system}]
    msgs.append({"role":"user","content": next((m.content for m in reversed(state['messages']) if isinstance(m, HumanMessage)), '')})
    msgs.append({"role":"assistant","content": state['messages'][-1].content})
    resp = llm.invoke(msgs)
    return {"messages":[AIMessage(content=resp.content)]}

from langgraph.graph import StateGraph, END
def build():
    g = StateGraph(State)
    g.add_node("plan", planner)
    g.add_node("fanout", fanout)
    g.add_node("synthesize", synthesize)
    g.set_entry_point("plan")
    g.add_edge("plan","fanout")
    g.add_edge("fanout","synthesize")
    g.add_edge("synthesize", END)
    return g.compile()

app = build()

def ask(q: str) -> str:
    state = {"messages":[HumanMessage(content=q)], "plan":[], "observations":{}}
    final=None
    for step in app.stream(state): final=step
    msgs = list(final.values())[0]["messages"]
    return msgs[-1].content

if __name__ == "__main__":
    print("ReAct Agent ready. Example questions:")
    print("- Who is Albert Einstein and when was he born?")
    print("- Compute 2^10 + sqrt(81)")
    print("- SELECT * FROM products WHERE category='Electronics'")
    while True:
        q=input("\n> ").strip()
        if q.lower() in {"exit","quit"}: break
        print(ask(q))
