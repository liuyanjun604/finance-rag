from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# call_llm 节点：输入是 state，返回更新后的 messages
def call_llm(state: MessagesState):
    # 用 llm_with_tools 调用，返回 {"messages": [response]}
    pass

# call_tools 节点：直接用 ToolNode
tool_node = ToolNode([get_stock_price])