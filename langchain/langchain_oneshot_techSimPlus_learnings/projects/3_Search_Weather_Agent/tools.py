from dotenv import load_dotenv
load_dotenv()
from langchain_nimble import NimbleSearchTool
from langchain.tools import tool
import os
import requests


search_tool = NimbleSearchTool(
    k = 5,
    deep_search = True,
    parsing_type="markdown"
)


@tool
def google_search(query:str):
    """
    Search anything on google for real time information.
    Args:
        query - user search query for google search
        
    Return - result from google search
    """
    res = search_tool.invoke(query)
    return res


@tool
def weather_tool(city:str):
    """
        Get real time weather details like temperature, humidity and others.
        Args:
            city - city name for weather details
        Return -> Weather data from the api response.
    """
    
    API_URL = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={os.getenv("WEATHER_API_KEY")}"
    res = requests.get(API_URL)
    if res.status_code == 200:
        return res.json()
    
    return "Unable to find details for this city"


ALL_TOOLS = [google_search, weather_tool]