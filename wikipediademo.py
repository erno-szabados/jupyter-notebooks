import requests

# TODO add and use requests-cache pip install requests-cache https://pypi.org/project/requests-cache/
# https://realpython.com/caching-external-api-requests/

url = 'https://hu.wikipedia.org/w/api.php?action=query'
params = {
    'titles': '',    
    'prop': 'extracts',
    'exlimit': '1',    
    'exintro': 'true',
    'exsentences': 5,
    'explaintext': '0',
    'exsectionformat': 'plain',
    'format': 'json',    
}

def wikiExtract(wordList):
    params['titles'] = "|".join(wordList)
    params['exlimit'] = len(wordList)
    result = {}
    response = requests.get(url=url, params=params)
    if response.ok:
        wiki_data = response.json()
        print(type(wiki_data['query']))
        for page_id, page_node in wiki_data['query']['pages'].items():
            title = page_node['title']
            if page_id != -1:
                result[title] = page_node['extract']
            else:
                result[title] = f'There is no data about {title}.'
    return result

if __name__ == "__main__":
    test_words = ['Pizza', 'Tészta']
    result = wikiExtract(test_words)
    print(result)