from flask import Flask, make_response, request
#from langchain_bot import *
import urllib.parse
import requests
import re
from bs4 import BeautifulSoup

# https://dagster.io/blog/chatgpt-langchain
# TODO: how can we make these install automatically?
# pip install langchain==0.0.55 requests openai transformers faiss-cpu

from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document

def constructUrl(baseUrl, path, categories=""):
    clientId = "c196d990-f964-457f-5690-552e7a52600c"
    # baseUrl = "https://acehelphub.humany.net/ace-help-hub-86/"
    site = "%2F%2Fopen-ai-testbed"
    funnel = "open-ai-testbed" #"ace-help-hub-86"
    qCat = ""
    if len(categories) > 0:
        qCat = "&categories=" + categories

    return baseUrl + path + "?client=" + clientId + "&funnel=" + funnel + "&site=" + site + qCat

def constructQueryUrl(phrase, interfaceBaseUrl, categories):
    return constructUrl(interfaceBaseUrl, "guides", categories) + "&phrase=" + urllib.parse.quote(phrase) + "&skip=0&take=6&sorting.type=popularity&sorting.direction=descending"
def constructGuideUrl(guideId, interfaceBaseUrl):
    return constructUrl(interfaceBaseUrl, "guides/" + guideId)

def getArticle(id, interfaceBaseUrl):
    url = constructGuideUrl(id, interfaceBaseUrl)
    resp = requests.post(url).json()
    # body = resp["Body"]
    body = BeautifulSoup(resp["Body"], "html.parser").get_text() #html5lib
    # sources.append(resp["Title"] + " \n" + body)
    return { "title": resp["Title"], "page_content": body, "metadata": {"source": url} }

def getRelevantSources(phrase, interfaceBaseUrl, categories):
    qUrl = constructQueryUrl(phrase, interfaceBaseUrl, categories)
    x = requests.post(qUrl)

    if x.status_code != 200:
        raise "Knowledge request error: " + x.reason
    
    totalLen = 0
    sources = []
    for match in x.json()["Matches"]:
        print(match["Title"])
        relativeUrl = match["RelativeUrl"]
        rx = re.search("^\d+", relativeUrl) 
        id = rx.group(0)
        
        article = getArticle(id, interfaceBaseUrl)
        content = article["page_content"]
        totalLen += len(content)
        if totalLen > 4000 / 1.1:
            break
        else:
            sources.append(Document(
                page_content=content,
                metadata=article["metadata"]
            )
    )
    return sources

def createAnswer(question, openAiApiKey, modelName, interfaceBaseUrl, categories, temperature):
    sources = getRelevantSources(question, interfaceBaseUrl, categories)
    chain = load_qa_with_sources_chain(
        OpenAI(temperature=temperature, openai_api_key=openAiApiKey,
        model_name=modelName # "text-davinci-003" # https://platform.openai.com/docs/models
        )
        # max_tokens=256 # sets an upper bound on how many tokens the API will return
    ) #, chain_type="map_reduce")
    
    try:
        return chain(
            {
                "input_documents": sources,
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    except Exception as e:
        return str(e)


app = Flask(__name__)

@app.route("/")
def hello():
    return app.send_static_file("index.html")

@app.route("/prompt")
def prompt():
    rPrompt = request.args.get("prompt")
    modelName = request.args.get("modelName")
    interfaceBaseUrl = request.args.get("interfaceBaseUrl")
    openAiApiKey = request.args.get("openAiApiKey")
    temperature = request.args.get("temperature")
    categories = request.args.get("categories")
    if not(rPrompt is None) and len(rPrompt):
        answer = createAnswer(rPrompt, openAiApiKey, modelName, interfaceBaseUrl, categories, float(temperature))
        response = make_response(answer, 200)
    else:
        response = make_response("prompt parameter empty", 200)

    response.mimetype = "text/plain"
    return response
