
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.conversation.base import ConversationChain
from langchain .prompts import PromptTemplate


#添加记忆模块
from langchain.memory import ConversationBufferMemory

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("api_key")
base_url = os.getenv("base_url")

online_llm = ChatOpenAI(
    api_key=api_key,
    model="qwen-plus",
    base_url=base_url,
    temperature=0
)


##初始化记忆模块
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#需要将记忆模块中的 关键 memory_key 添加到提示词中，大模型对话时无法使用记忆内容
template = '''
你是一个全能助手，可以根据用户的问题给出合适的答案。对话历史如下：{chat_history}。问题如下：{question}
'''

prompt = PromptTemplate(
    input_variables=['chat_history',"question"],
    template=template
)

#定义对话链
llm_chain = ConversationChain(
    llm=online_llm,
    prompt=prompt,
    memory=memory,
    verbose=True,
    input_key='question'
)


result = llm_chain.invoke("你是谁？")
print(result)

print('第一次问题后的对话历史：',memory.load_memory_variables({}))

result2 = llm_chain.invoke("我的上一个问题是什么？")
print(result2)

print('第二次问题后的对话历史：',memory.load_memory_variables({}))

# result3 = llm_chain.invoke("llm是什么？")
# print(result3['text'])

# print('第三次问题后的对话历史：',memory.load_memory_variables({}))
