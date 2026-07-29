from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

def ice_break_with(name: str) -> str:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)

    summary_template = """
        given the Linkedin information {information} about a person I want you to create:
        1. A short summary
        2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0
    )

    chain = summary_prompt_template | llm

    res = chain.invoke({"information": linkedin_data})

    print(res.content)

if __name__ == "__main__":
    load_dotenv()
    
    print("Hello Pratham")

    ice_break_with(name="Emmanuel Macron")
