import os
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_agent
from langchain_classic import hub
from tools.tools import get_profile_url_tavily

def lookup(name: str) -> str:
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0
    )

    template = """gien the full name {name_of_person} I want you to get me a link to their linkedin profile page. You answer should contain only a URL"""

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["name_of_person"]
    )
    
    tools_for_agent = [
        Tool(
            name="Crawl Google for linkedin profile page",
            func=get_profile_url_tavily,
            description="Useful for when you need to get a linkedin profile URL"
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_agent(
        model=llm,
        tools=tools_for_agent,
        prompt=react_prompt
    )

    result = agent.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linkedin_profile_url = result["output"]

    return linkedin_profile_url