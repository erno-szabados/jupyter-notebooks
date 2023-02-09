import requests
import torch
from transformers import pipeline
from nltk.stem import WordNetLemmatizer
import gradio as gr


# TODO add and use requests-cache pip install requests-cache https://pypi.org/project/requests-cache/
# https://realpython.com/caching-external-api-requests/


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    device_id = torch.cuda.current_device()

#ner_mname = "NYTK/named-entity-recognition-nerkor-hubert-hungarian"
ner_mname = "dslim/bert-base-NER"
ner_pipeline = pipeline(task="ner", model=ner_mname, device=device_id)
lemmatizer = WordNetLemmatizer()


def wikiExtract(word_list, lang='en'):
    url = f'https://{lang}.wikipedia.org/w/api.php?action=query'
    params = {
        'titles': '',    
        'prop': 'extracts',
        'exlimit': '1',    
        'exintro': 'true',
        'exsentences': 5,
        'explaintext': '0',
        'exsectionformat': 'plain',
        'format': 'json',    
        'redirects': True
    }
    params['titles'] = "|".join(word_list)
    params['exlimit'] = len(word_list)
    result = {}
    response = requests.get(url=url, params=params)
    if response.ok:
        wiki_data = response.json()
        for page_id, page_node in wiki_data['query']['pages'].items():
            title = page_node['title']
            if int(page_id) > 0:
                result[title] = page_node['extract']
            else:
                result[title] = f'No data about {title}.'
    return result

def get_text_ner_defs(input_text):
    response = ner_pipeline(input_text, aggregation_strategy="max")
    word_list = [item["word"] for item in response if item['score'] > 0.8]
    stem_words = [lemmatizer.lemmatize(w) for w in set(word_list)]
    result = wikiExtract(stem_words)
    return result

def process(input_text):   
    result = get_text_ner_defs(input_text=input_text)
    t = [(v,k) for k, v in result.items()]
    return t

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("Wikipedia Annotator searches for named entities in wikipedia and shows info about them.")
        with gr.Row():
            input_text = gr.Textbox(placeholder="Enter text to annotate")
            output_text = gr.HighlightedText(label="Annotations")
        btn = gr.Button("Annotate")
        btn.click(fn=process, inputs=input_text, outputs=output_text)

    demo.title = "Wikipedia Annotator"
    demo.launch()