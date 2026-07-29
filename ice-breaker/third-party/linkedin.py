import os
import requests
from dotenv import load_dotenv

load_dotenv()

def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False):
    """
    Scrape information from linkedin profiles, manually scrape the information from linkedin profile
    """
    if mock:
        linkedin_profile_url = ""
        response = requests.get(linkedin_profile_url, timeout=10)
        return response.json()
    else:
        api_endpoint = "https://api.scrapin.io/enrichment/profile"
        params = {
            "api_key": os.getenv("SCRAPIN_API_KEY"),
            "url": linkedin_profile_url
        }
        response = requests.get(api_endpoint, params=params, timeout=10)
    
    data = response.json().get("person")
    return data


if __name__ == "__main__":
    print(scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/pratham-jain-dev/", mock=True
    ))