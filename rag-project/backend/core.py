from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings,ChatOpenAI

INDEX_NAME = "langchain_doc_index"

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    chat = ChatOpenAI(verbose=True, temperature=0)
    retrieval_qa_chat_chain = hub.pull("langchain-ai/retrieval-qa-chat")

    rephrase_prompt = hub.pull("langchain-ai/chain-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat,
        retriever=docsearch.as_retriever(),
        prompt=rephrase_prompt
    )
    
    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=retrieval_qa_chat_chain
    )
    
    result = qa.invoke({"input": query})
    new_result = {
        "query": result["input"], 
        "result": result["answer"],
        "source_documents": result["context"]
    }
    
    return new_result

if __name__ == "__main__":
    res = run_llm(query="what is Langchain Chain?")
    print(res["answer"])