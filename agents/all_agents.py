# from dotenv import load_dotenv

# from dotenv import load_dotenv
# from langchain import hub
# from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
# from all_tools import tools, dm_tools,dm_group_tools
# load_dotenv()

# # llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash-exp", temperature = 0.7, max_retries=1)
# # llm = ChatGoogleGenerativeAI(model = "gemini-1.5-pro", temperature = 0.7)

# llm = ChatOpenAI(model='gpt-4o', temperature=0.6, max_retries=1)

# # Pull the prompt template from the hub
# # prompt = hub.pull("hwchase17/openai-tools-agent")
# from prompt import custom_prompt,direct_message_prompt,direct_group_message_prompt
# # Create the ReAct agent using the create_tool_calling_agent function
# agent = create_tool_calling_agent(
#     llm=llm,
#     tools=tools,
#     prompt=custom_prompt,
# )

# # Create the agent executor
# agent_exec = AgentExecutor.from_agent_and_tools(
#     agent=agent,
#     tools=tools,
#     verbose=True,
#     handle_parsing_errors=True,
#     # max_iterations=5,
# )

# dm_agent = create_tool_calling_agent(
#     llm=llm,
#     tools=dm_tools,
#     prompt=direct_message_prompt,
# )

# # Create the agent executor
# dm_agent_exec = AgentExecutor.from_agent_and_tools(
#     agent=dm_agent,
#     tools=dm_tools,
#     verbose=True,
#     handle_parsing_errors=True,
#     # max_iterations=5,
# )


# dm_group_agent = create_tool_calling_agent(
#     llm=llm,
#     tools=dm_group_tools,
#     prompt=direct_group_message_prompt,
# )

# # Create the agent executor
# dm_group_agent_exec = AgentExecutor.from_agent_and_tools(
#     agent=dm_group_agent,
#     tools=dm_group_tools,
#     verbose=True,
#     # max_iterations=5,
# )
# # Test the agent with sample queries
# # response = agent_executor.invoke({"input": "Search for Apple Intelligence"})
# # print("Response for 'Search for LangChain updates':", response)

# # response = agent_executor.invoke({"input": "Multiply 10 and 20"})
# # print("Response for 'Multiply 10 and 20':", response)
# agents.py
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from prompt import schedule_prompt, update_prompt, delete_prompt,calender_prompt,schedule_group_prompt,update_group_prompt,schedule_channel_prompt

load_dotenv()

# Initialize the language model
# llm = ChatOpenAI(model='gpt-4o-mini',temperature=0, max_retries=1)
llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash-exp", temperature = 0.7, max_retries=1)

def create_schedule_agent(tools):
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=schedule_prompt,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

def create_schedule_group_agent(tools):
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=schedule_group_prompt,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
def create_schedule_channel_agent(tools):
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=schedule_channel_prompt,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
def create_update_group_agent(tools):
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=update_group_prompt,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
def create_calendar_agent(tools):
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=calender_prompt,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

def create_update_agent(tools):
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=update_prompt,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

def create_delete_agent(tools):
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=delete_prompt,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
