import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from ai-course!")
    print(os.environ.get("OPENAI_API_KEY"))
    print(f"Google API Key: {os.environ.get('GOOGLE_API_KEY')}")

    information = """
    This is an AI course that covers various topics in artificial intelligence, including machine learning, natural language processing, computer vision, and more. The course is designed to provide students with a comprehensive understanding of AI concepts and techniques, as well as practical skills for implementing AI solutions. Students will learn about the latest advancements in AI research and applications, and will have the opportunity to work on real-world projects to apply their knowledge.
    """

    summary_template ="""Summarize the following information: {information}"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm = ChatOllama(temperature=0, model="ollama")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)



if __name__ == "__main__":
    main()
