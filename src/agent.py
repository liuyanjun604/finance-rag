from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from tool_calling import get_stock_price
from dotenv import load_dotenv
import os
from config import BASE_URL, LLM_MODEL

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=BASE_URL,
    model=LLM_MODEL
)

# call_llm 节点：输入是 state，返回更新后的 messages
def call_llm(state: MessagesState):
    # 用 llm_with_tools 调用，返回 {"messages": [response]}
    llm_with_tools = llm.bind_tools([get_stock_price])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# call_tools 节点：直接用 ToolNode
tool_node = ToolNode([get_stock_price])

graph = StateGraph(MessagesState)
graph.add_node("call_llm", call_llm)
graph.add_node("tools", tool_node)
graph.add_edge(START, "call_llm")
graph.add_conditional_edges("call_llm", tools_condition)
graph.add_edge("tools", "call_llm")
app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "messages": [{"role": "user", "content": "花旗银行股票代码是C，现在股价多少？"}]
    })
    print(result["messages"][-1].content)