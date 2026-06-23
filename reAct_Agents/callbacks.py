from typing import Any, Dict, List
from uuid import UUID

from langchain_classic.schema import LLMResult
from langchain_core.callbacks import BaseCallbackHandler

class AgentCallbackHandler(BaseCallbackHandler):
    """"
    Basically we can inherit and override the functions that will be triggered in every Langchain Interesting Event.
    Q. What are the interesting events?
    Ans. Maybe a call to the LLM, a response from LLM, when we select a tool, after we execute a tool, when we have an error, when we have a new token on the LLM
    """

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        print(f"***Prompt to the LLM was:***\n{prompts[0]}")
        print("********")
        return


    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print(f"***LLM Response:***\n{response.generations[0][0].text}")
        print("********")
        return
