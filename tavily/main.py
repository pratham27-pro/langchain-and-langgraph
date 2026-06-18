from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import create_agent
# from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama

from prompt import REACT_PROMPT_WITH_EXACT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]
llm = ChatOllama(model="ollama")
structured_llm = llm.with_structured_output(AgentResponse)
react_prompt = hub.pull("hwchase17/react")
# output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_exact_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_EXACT_INSTRUCTIONS,
    input_variables=["tools", "input", "agent_scratchpad"]
# ).partial(format_instructions=output_parser.get_format_instructions())
).partial(format_instructions="")

agent = create_agent(model=llm, tools=tools, system_prompt=react_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
# parse_output = RunnableLambda(lambda x: output_parser.parse(x))

# chain = agent_executor | extract_output | parse_output
chain = agent_executor | extract_output | structured_llm

def main():
    result = chain.invoke(
        input={
            "input" : "Search for 3 job postings for an AI engineer using langchain in the bay area on Linkedin and list their details"
        }
    )
    print(result)

