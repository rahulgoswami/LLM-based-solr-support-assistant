#!/usr/bin/env python3
# scripts/agent_service.py

import os
from dotenv import load_dotenv

# 1. Load .env
load_dotenv()

# 2. New import for ChatOpenAI
from langchain_openai import ChatOpenAI

# 3. Pull in the default ReAct prompt template from LangChain Hub
from langchain import hub

# 4. Agent & execution imports
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor

# 5. Your tools
from agent_tools import DocRetriever, LogSearcher, ConfigValidator, Summarizer

# ——— CONFIG ———
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your .env")

# 6. Instantiate the LLM (no more deprecation warning)
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.0,
    openai_api_key=OPENAI_API_KEY
)

# 7. Load the default ReAct prompt so prompt is never None
#    This is the same template used by create_react_agent if you omit a prompt.
prompt = hub.pull("hwchase17/react")  # LangChain Hub ID for the ReAct template

# 8. Initialize tools (you already fixed Pydantic issues in agent_tools.py)
tools = [
    DocRetriever(persist_dir="vector_store", collection_name="solr_support", embed_model="all-mpnet-base-v2"),
    #LogSearcher(log_dir="/var/log/solr"),
    LogSearcher(log_dir="/Users/rahulgoswami/Desktop/Lab/log/solr"),
    ConfigValidator(),
    Summarizer(llm)
]

# 9. Create the ReAct agent with a real prompt object
react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# 10. Wrap in an executor
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

def run_agent(query: str) -> str:
    # Pass your query under the 'input' key (AgentExecutor.input_keys == ['input'])
    response = agent_executor.invoke({"input": query})
    answer   = response["output"]
    # If you need the reasoning steps, you can also do:
    # steps = response["intermediate_steps"]
    return answer


if __name__ == "__main__":
    q = input("Enter your Solr question: ")
    print(run_agent(q))

