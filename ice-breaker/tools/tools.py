from langchain_tavily import TavilySearch

def get_profile_url_tavily(name: str):
    """Searches fro linkedin or twitter profile page"""
    search = TavilySearch()
    res = search.run(f"{name}")
    return res