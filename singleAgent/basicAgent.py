

import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
import datetime
import requests

load_dotenv()

#获取环境变量
api_key = os.getenv('api_key')
base_url = os.getenv('base_url')

# 聚合网相关api调用-根据城市查询天气 - 代码参考（根据实际业务情况修改）
# 基本参数配置
apiUrl = 'http://apis.juhe.cn/simpleWeather/query'  # 接口请求URL
juheAppKey = os.getenv('juheAppKey')  # 在个人中心->我的数据,接口名称上方查看


online_llm = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    temperature=0,
    model='qwen-plus'
)

# result = online_llm.invoke("你是谁？")
# print(result.content)



def get_weather(city:str):
    # 接口请求入参配置
    requestParams = {
        'key': juheAppKey,
        'city': city,
    }
  # 发起接口网络请求
    response = requests.get(apiUrl, params=requestParams)

    # 解析响应结果
    if response.status_code == 200:
        responseResult = response.json()
        # 网络请求成功。可依据业务逻辑和接口文档说明自行处理。
        print(responseResult)
        return responseResult
    else:
        # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
        return '请求异常'







def getDate(_):
    '''获取当前日期'''

    return datetime.date.today()

# print(getDate(None))


# def get_weather(location: str) -> str:
#     # 这里我们可以调用一个天气API，或者返回假数据
#     return f"假设{location}的天气是晴天，温度25°C"


# 将Tool添加到智能体中
tools = [
    Tool(
        name="WeatherAPI", 
        func=get_weather, 
        description="可以根据城市名字，查询一个城市的天气情况"
    ),
    Tool(
        name='getDate',
        func=getDate,
        description="获取当前日期"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=online_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

response = agent.run("请告诉我今天和后天的日期，以及明天北京的天气情况")
print(response)


