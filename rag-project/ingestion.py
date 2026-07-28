import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_info, log_warning, log_success, log_header)

load_dotenv()


# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10
)

# chroma = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=embeddings
# )

vector_store = PineconeVectorStore(
    index_name="langchain_docs_2025",
    embedding=embeddings
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_docs_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously"""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"Vectorstore indexing: Preparing to add {len(documents)} documents to vector store...",
        Colors.DARKCYAN
    )

    # create batches
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    log_info(f"Created {len(batches)} batches of documents", Colors.DARKCYAN)

    # process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vector_store.aadd_documents(batch)
            log_success(f"Batch {batch_num} added successfully")
        except Exception as e:
            log_error(f"Failed to add batch {batch_num}: {e}")
            return False
        return True

    tasks = [add_batch(batch, i) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # count successful batches
    successful_batches = sum(1 for result in results if result is True)

    if successful_batches == len(batches):
        log_success(f"All {len(batches)} batches added successfully")
    else:
        log_warning(f"Only {successful_batches}/{len(batches)} batches added successfully")


async def main():
    """Main async function to orchestrate the entire process"""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info("TavilyMap: Mapping documentation site...", Colors.PURPLE)

    # map the site first to get list of URLs
    site_map = await tavily_map.ainvoke({"url": "https://python.langchain.com/"})
    log_success(f"TavilyMap: Successfully mapped {len(site_map['results'])} URLs")

    log_info("TavilyCrawl: Starting to crawl documentation...", Colors.PURPLE)

    # crawl documentation site
    res = await tavily_crawl.ainvoke({
        "url": "https://python.langchain.com/",
        "max_depth": 5,
        "extract_depth": "advanced",
        "instructions": "content on AI agents"
    })
    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in res["results"]
    ]
    log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site")

    # split docs into chunks
    log_header("DOC CHUNKING PHASE")
    log_info("RecursiveCharacterTextSplitter: Splitting documents into chunks...", Colors.YELLOW)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"RecursiveCharacterTextSplitter: Successfully split documents into {len(splitted_docs)} chunks")

    # process docs async
    await index_docs_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETED")
    log_success("Documentation ingestion pipeline completed successfully!")
    log_info("Summary", Colors.BOLD)
    log_info(f"URLs mapped: {len(site_map['results'])}")
    log_info(f"Documents extracted: {len(all_docs)}")
    log_info(f"Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())