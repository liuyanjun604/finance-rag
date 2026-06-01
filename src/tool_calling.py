from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from fastapi import FastAPI
import os
from config import BASE_URL, LLM_MODEL
from langchain_core.messages import HumanMessage, ToolMessage
import yfinance as yf

app = FastAPI()
load_dotenv() 
llm = ChatOpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=BASE_URL,
    model=LLM_MODEL
)

# 第一步：定义工具，用 @tool 装饰器
@tool
def get_stock_price(ticker: str) -> str:
    """查询股票价格，输入股票代码，返回当前价格。"""
    try:
        stock = yf.Ticker(ticker)
        price = stock.fast_info.last_price
        if price is None:
            return f"未能获取 {ticker} 的股价，请稍后再试"
        return f"{ticker} 当前股价为 ${price:.2f}"
    except Exception as e:
        return f"查询失败：{str(e)}"

# 第二步：把工具绑定给 LLM
llm_with_tools = llm.bind_tools([get_stock_price])

# 第三步：用户提问，LLM 自动判断要不要调用工具
if __name__ == "__main__":
    response = llm_with_tools.invoke("花旗银行的股票代码是C，现在股价多少？")

    #第四步：把结果塞回LLM
    messages = [
        HumanMessage(content="花旗银行的股票代码是C，现在股价多少？"),
        response,  # LLM 的第一轮回复（包含 tool_call）
    ]

    # 执行每个工具调用
    for tool_call in response.tool_calls:
        tool_result = get_stock_price.invoke(tool_call["args"])
        messages.append(ToolMessage(
            content=tool_result,
            tool_call_id=tool_call["id"]
        ))

    # 第五步：LLM 拿到工具结果，组织最终回答
    final_response = llm_with_tools.invoke(messages)
    print(final_response.content)