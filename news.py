import requests
import urllib.parse
import json
API_KEY = "0c6455b3-21b8-45f1-bed7-09857f7f1f66"


r = requests.get('https://api.goperigon.com/v1/all', params={"apiKey": API_KEY})


print(type(r.json()))

import requests




def getCurrentNews():
    url = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/search/NewsSearchAPI"
    headers = {
        "X-RapidAPI-Key": "7fb876ddc9msh3c2b8e24a8f4aabp14d92djsn23126e2798c1",
        "X-RapidAPI-Host": "contextualwebsearch-websearch-v1.p.rapidapi.com"
    }
    querystring = {"q":"Current News","pageNumber":"1","pageSize":"10","autoCorrect":"true","fromPublishedDate":"null","toPublishedDate":"null"}

    
    response = requests.request("GET", url, headers=headers, params=querystring).json()
    return response
