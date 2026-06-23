from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic.schema import AgentAction, AgentFinish
from langchain_core.tools.render import render_text_description
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_protocol import Union

from callbacks import AgentCallbackHandler
from reAct_Agents.log import format_log_to_str

load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """Get the length of a text string."""
    print(f"get_text_length entered with {text=}")
    text = text.strip("'\n").strip('"')

    # stripping away non-alphabetic chars just in case

    return len(text)

def find_tool_by_name(tools, name):
    for tool in tools:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool with name {name} not found")

if __name__ == "__main__":
    print("Hello langchain!")
    tools = [get_text_length]
    print(get_text_length.invoke(input={"text": "dog"}))

# Note we can't directly call the get_text_length function because it is wrapped in a tool decorator. Instead, we can use the invoke method to call it with the required input.

    template="""
    Answer the following questions as best you can. You have access to the following tools:
    {tools}
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    Begin!
    Question: {input}
    Thought:{agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([tool.name for tool in tools])
    )

    llm = ChatOpenAI(temperature=0, callbacks=[AgentCallbackHandler()]).bind(stop=["\nObservation", "Observation"])
    intermediate_steps = []
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )
    # Here the pipe operator is part of Langchain Expression Language, it takes the output of the left side and plugs it into the input of its right side.

    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] =agent.invoke(
            {
                "input": "What is the length of the text 'dog'?",
                "agent_scratchpad": intermediate_steps
            }
        )
        print(agent_step)

        # res = agent.invoke(input={"input": "What is the length of the text 'dog'?)"})
        # print(res)

        if isinstance(agent_step, AgentFinish):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            print(f"Observation: {observation}")

