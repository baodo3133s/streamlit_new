import os
from typing import List

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langgraph.prebuilt import create_react_agent


# ============================================================
# 1. Environment Setup
# ============================================================

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key=API_KEY
)


# ============================================================
# 2. PDF Loader + Vector Store Builder
# ============================================================

PDF_FOLDER = "pdfs"


def load_all_pdfs(folder_path: str):
    """
    Load all PDF files from a folder and return a list of LangChain documents.
    """
    docs = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            print(f"[INFO] Loading PDF: {full_path}")
            loader = PyPDFLoader(full_path)
            docs.extend(loader.load())
    return docs


def build_retriever(folder_path: str):
    """
    Build a retriever from all PDFs in the folder.
    """
    docs = load_all_pdfs(folder_path)

    if not docs:
        raise ValueError(f"No PDFs found in folder: {folder_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(api_key=API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever()


retriever = build_retriever(PDF_FOLDER)


# ============================================================
# 3. Tools
# ============================================================

@tool
def search_project_docs(query: str) -> str:
    """Search the project PDF knowledge base for relevant information."""
    results = retriever.invoke(query)

    if not results:
        return "No relevant content found in the project documents."

    output = []
    for i, doc in enumerate(results, start=1):
        meta = doc.metadata or {}
        page = meta.get("page", "N/A")
        source = meta.get("source", "unknown.pdf")

        header = f"[Result {i} | {source} | page {page}]"
        output.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n".join(output)



TOOLS = [search_project_docs]


# ============================================================
# 4. System Prompt (Project Manager Persona)
# ============================================================

SYSTEM_MESSAGE = """
You are PM‑Assist, an expert project manager AI.

You use the project PDFs as your knowledge base. These may include:
- project charters
- requirements documents
- timelines
- risk registers
- stakeholder analyses
- status reports

Your responsibilities:
- Retrieve information using the search_project_docs tool.
- Provide structured, actionable project management guidance.
- When answering, organize content into sections such as:
  Overview, Details, Risks, Dependencies, Recommendations, Next Steps.
- If the user asks about something not covered in the PDFs, say so clearly,
  then provide best‑practice PM guidance.

Always call search_project_docs before giving a substantive answer.
"""


# ============================================================
# 5. LangGraph Agent
# ============================================================

agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_MESSAGE)


def run_agent(user_input: str, history: List[BaseMessage]) -> AIMessage:
    """
    Run a single turn of the agent with LangGraph tool execution.
    """
    try:
        result = agent.invoke(
            {"messages": history + [HumanMessage(content=user_input)]},
            config={"recursion_limit": 50}
        )
        return result["messages"][-1]
    except Exception as e:
        return AIMessage(
            content=f"Error: {str(e)}\n\nCheck your PDFs or try rephrasing."
        )



