from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, json, logging, httpx, re
from datetime import datetime
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict
import google.generativeai as genai
import html

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="Math Routing Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in .env file")

genai.configure(api_key=GOOGLE_API_KEY)

#MODELS 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.2,
    max_output_tokens=2048,
    google_api_key=GOOGLE_API_KEY
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)


class MathQuery(BaseModel):
    question: str
    user_id: Optional[str] = "anonymous"

class FeedbackRequest(BaseModel):
    query_id: str
    rating: int
    feedback_text: Optional[str] = None
    user_id: Optional[str] = "anonymous"

class MathResponse(BaseModel):
    query_id: str
    question: str
    solution: str
    steps: List[str]
    source: str
    confidence: float
    timestamp: str
    route_log: List[str]

class InputGuardrail:
    MATH_KEYWORDS = [
        'solve', 'calculate', 'prove', 'derivative', 'integral', 'equation',
        'theorem', 'geometry', 'algebra', 'limit', 'matrix', 'vector', 'series'
    ]

    @staticmethod
    def validate(query: str) -> tuple[bool, str]:
        q = query.lower()
        if any(bad in q for bad in ["hack", "illegal", "exploit"]):
            return False, "Inappropriate content"
        if any(k in q for k in InputGuardrail.MATH_KEYWORDS) or any(sym in q for sym in "+-*/=^∫π"):
            return True, "Valid math query"
        return False, "Not a math question"

class OutputGuardrail:
    @staticmethod
    def validate(resp: str) -> tuple[bool, str]:
        if not resp or len(resp.strip()) < 10:
            return False, "Response too short"
        return True, "Valid output"

#KNOWLEDGE BASE
class KnowledgeBase:
    def __init__(self):
        self.db_path = "knowledge_base/chroma_db"
        os.makedirs("knowledge_base", exist_ok=True)
        self.vectorstore = self._init_db()

    def _init_db(self):
        docs = []
        for f in os.listdir("knowledge_base"):
            if f.endswith(".txt"):
                with open(os.path.join("knowledge_base", f), "r") as file:
                    docs.append(file.read())

        if not docs:
            sample = [
                "Q: Derivative of x^2?\nA: 2x\nSteps: power rule.",
                "Q: Solve x^2 - 5x + 6 = 0\nA: x=2 or 3\nSteps: factorization."
            ]
            with open("knowledge_base/default_math.txt", "w") as f:
                f.write("\n\n".join(sample))
            docs = sample

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.create_documents(docs)

        if os.path.exists(self.db_path):
            logger.info("Loading existing Chroma DB...")
            return Chroma(persist_directory=self.db_path, embedding_function=embeddings)
        else:
            logger.info("Creating new Chroma DB...")
            db = Chroma.from_documents(texts, embedding=embeddings, persist_directory=self.db_path)
            db.persist()
            return db

    def search(self, query: str, k: int = 3):
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            results = retriever.invoke(query)
            if not results:
                return []
            return [{"content": r.page_content, "score": 0.9} for r in results]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

kb = KnowledgeBase()
#WEB SEARCH
class WebSearchMCP:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.tavily.com/search"

    async def search(self, query: str):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.url,
                    json={"api_key": self.api_key, "query": f"math: {query}", "max_results": 3},
                    timeout=15
                )
                data = resp.json() if resp.status_code == 200 else {}
                if not data.get("results"):
                    logger.warning("No Tavily results found.")
                    return {"success": False, "results": [], "answer": ""}
                return {
                    "success": True,
                    "results": data.get("results", []),
                    "answer": data.get("answer", "") or "No direct answer found. Use context from results."
                }
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {"success": False, "error": str(e)}

web_search = WebSearchMCP(TAVILY_API_KEY)

class AgentState(TypedDict):
    question: str
    guardrail_passed: bool
    kb_results: List[Dict]
    web_results: Dict
    solution: str
    steps: List[str]
    source: str
    confidence: float
    route_log: List[str]


import html

def clean_json_string(s: str) -> str:
    
    if not s:
        return ""

    s = html.unescape(s)

    s = re.sub(r"[\x00-\x1F\x7F]", "", s)

    s = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

    s = s.replace("“", '"').replace("”", '"').replace("’", "'")

    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s.strip()


def create_agent():
    workflow = StateGraph(AgentState)

    def input_guardrail_node(state: AgentState):
        valid, msg = InputGuardrail.validate(state["question"])
        state["guardrail_passed"] = valid
        state["route_log"].append(msg)
        logger.info(msg)
        return state

    def kb_search_node(state: AgentState):
        if not state["guardrail_passed"]:
            state["route_log"].append("Guardrail failed")
            return state
        results = kb.search(state["question"])
        state["kb_results"] = results
        if results:
            logger.info("Found relevant KB results")
            state["route_log"].append("📚 KB hit")
        else:
            logger.info("KB miss — fallback to Web Search")
            state["route_log"].append("KB miss → Web Search")
        return state

    def router(state: AgentState) -> str:
        if state["kb_results"]:
            state["source"] = "knowledge_base"
            state["confidence"] = 0.9
            logger.info("Using Knowledge Base")
            return "generate"
        else:
            state["source"] = "web_search"
            state["confidence"] = 0.6
            logger.info("Using Web Search")
            return "web_search"

    async def web_node(state: AgentState):
        results = await web_search.search(state["question"])
        state["web_results"] = results
        if results.get("success"):
            state["route_log"].append("Web search success")
        else:
            state["route_log"].append("Web search failed")
        return state


    async def generate_node(state: AgentState, retries: int = 0):
        if retries > 2:
            logger.warning("Stopping retries after 2 failed attempts.")
            state["solution"] = "Could not generate valid output after multiple attempts."
            state["steps"] = ["LLM failed repeatedly."]
            state["confidence"] = 0.2
            return state

        context = ""
        if state["source"] == "knowledge_base" and state["kb_results"]:
            context = "\n".join([r["content"] for r in state["kb_results"][:2]])
        elif state["source"] == "web_search" and state.get("web_results", {}).get("success"):
            web_data = state["web_results"]
            snippets = [r.get("content") or r.get("snippet", "") for r in web_data.get("results", [])]
            context = web_data.get("answer", "") + "\n".join(snippets)
        else:
            context = "No relevant context found."

        prompt = f"""
    You are a mathematics reasoning engine.
    Use the provided context (which may contain partial solutions or formulas) to solve the problem step by step.

    Question: {state['question']}
    Context: {context}

    Return valid JSON only in this format:
    {{
    "solution": "Final answer in plain text",
    "steps": ["Step 1...", "Step 2...", "Step 3..."]
    }}
    """.strip()

        try:
            resp = llm.invoke([HumanMessage(content=prompt)], response_mime_type="application/json")
            result = json.loads(resp.content)
        except Exception as e:
            err = str(e)

            # Web search fallback for invalid output
            if any(x in err for x in ["429", "ResourceExhausted", "Invalid", "Unterminated"]):
                logger.warning("Gemini output invalid → switching to Web Search")
                state["route_log"].append("Gemini parse failed → Web Search fallback")
                state["source"] = "web_search"
                state["confidence"] = 0.4
                web_results = await web_search.search(state["question"])
                state["web_results"] = web_results
                return await generate_node(state, retries + 1)

            if state["source"] == "web_search" and state.get("web_results", {}).get("results"):
                web_text = "\n".join(
                    [r.get("content") or r.get("snippet", "") for r in state["web_results"]["results"]]
                )
                try:
                    small_prompt = f"Extract or compute the most likely math answer (step-by-step) for:\n{state['question']}\n\nContext:\n{web_text}\n\nAnswer in JSON {{'solution': '...', 'steps': [...]}}"
                    resp2 = llm.invoke([HumanMessage(content=small_prompt)], response_mime_type="application/json")
                    result = json.loads(resp2.content)
                except Exception:
                    result = {
                        "solution": "Could not derive correct answer, even after web fallback.",
                        "steps": ["Web results did not yield a computable solution."]
                    }
            else:
                result = {
                    "solution": "Could not parse JSON. Model output was invalid.",
                    "steps": ["Fallback used because Gemini output failed."]
                }

        state["solution"] = result.get("solution", "No valid solution generated.")
        state["steps"] = result.get("steps", ["No clear steps were found."])
        state["route_log"].append(f"Using {state['source']}")
        return state






    workflow.add_node("input", input_guardrail_node)
    workflow.add_node("kb", kb_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("web_search", web_node)

    workflow.set_entry_point("input")
    workflow.add_edge("input", "kb")
    workflow.add_conditional_edges("kb", router, {"web_search": "web_search", "generate": "generate"})
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()

math_agent = create_agent()

@app.post("/query", response_model=MathResponse)
async def query_math(q: MathQuery):
    try:
        init_state = {
            "question": q.question,
            "guardrail_passed": False,
            "kb_results": [],
            "web_results": {},
            "solution": "",
            "steps": [],
            "source": "",
            "confidence": 0.0,
            "route_log": []
        }

        result = await math_agent.ainvoke(init_state)
        valid, _ = OutputGuardrail.validate(result.get("solution", ""))

        return MathResponse(
            query_id=f"q_{datetime.now().timestamp()}",
            question=q.question,
            solution=result.get("solution", "No response"),
            steps=result.get("steps", []),
            source=result.get("source", "unknown"),
            confidence=result.get("confidence", 0.0),
            timestamp=datetime.now().isoformat(),
            route_log=result.get("route_log", [])
        )

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Math Routing Agent (KB + Web Search + Logs)", "version": "2.3"}

feedback_store = []

@app.post("/feedback")
async def feedback(data: FeedbackRequest):
    feedback_store.append({**data.dict(), "timestamp": datetime.now().isoformat()})
    logger.info(f"Stored feedback for {data.query_id}")
    return {"status": "ok", "message": "Feedback saved"}

@app.get("/health")
async def health():
    return {"status": "healthy", "feedback_count": len(feedback_store)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
