from flask import Flask, make_response, request
#from langchain_bot import *
import urllib.parse
import requests
import re
from bs4 import BeautifulSoup
import json
import os
# https://dagster.io/blog/chatgpt-langchain
# TODO: how can we make these install automatically?
# pip install langchain==0.0.55 requests openai transformers faiss-cpu

from langchain.llms import OpenAI, AzureOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01"
os.environ["OPENAI_API_BASE"] = "https://open-ai-ace-test.openai.azure.com"
# The API key for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
#export OPENAI_API_KEY=<your Azure OpenAI API key>

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
            sources.append(
                {
                    "content": content,
                    "metadata": article["metadata"]
                }
                # Document(
                #     page_content=content,
                #     metadata=article["metadata"]
                # )
            )
    return sources

def azureOpenAiCompletion(prompt, api_key, temperature = 0):
    base_url = "https://open-ai-ace-test.openai.azure.com/"
    deployment_name ="text-davinci-003-test"

    url = base_url + "/openai/deployments/" + deployment_name + "/completions?api-version=2022-12-01"
    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference
    payload = {
        "prompt":prompt,
        "temperature": temperature,
        "max_tokens": 40
    }

    r = requests.post(url, 
        headers={
            "api-key": api_key,
            "Content-Type": "application/json"
        },
        json = payload
    )

    response = json.loads(r.text)
    formatted_response = json.dumps(response, indent=4)
    print(formatted_response)

    return response["choices"][0]["text"]

def openAiCompletion(question, sources, apiKey, modelName, temperature = 0):
    # https://langchain.readthedocs.io/en/latest/modules/llms/integrations/azure_openai_example.html
    chain = load_qa_with_sources_chain(
        AzureOpenAI(deployment_name=modelName, model_name="text-davinci-003", temperature=temperature, openai_api_key=apiKey)
        # OpenAI(temperature=temperature, openai_api_key=apiKey, model_name=modelName)
        # "text-davinci-003" # https://platform.openai.com/docs/models
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

# def createAnswer(question, apiKey, modelName, interfaceBaseUrl, categories, temperature):
#     sources = getRelevantSources(question, interfaceBaseUrl, categories)


##result = azureOpenAiCompletion("What happened to Elmo?", "a2a5db8d23324cb6a937e8ab119be9ca", 0.5)
##print(result)
#sources = getRelevantSources("Where is my latest invoice?", interfaceBaseUrl, categories)
#answer = openAiCompletion(rPrompt, sources, apiKey, modelName, float(temperature)) #"text-davinci-003"



app = Flask(__name__)

@app.route("/")
def hello():
    return app.send_static_file("index.html")

@app.route("/step1")
def prompt():
    rPrompt = request.args.get("prompt")
    interfaceBaseUrl = request.args.get("interfaceBaseUrl")
    categories = request.args.get("categories")
    if not(rPrompt is None) and len(rPrompt):
        sources = getRelevantSources(rPrompt, interfaceBaseUrl, categories)
        # answer = createAnswer(rPrompt, apiKey, modelName, interfaceBaseUrl, categories, float(temperature))
        #xxx = list(map(lambda src: { content : src.page_content, metadata : src.metadata }, sources))
        # page_content=content, metadata=article["metadata"]

        asJson = json.dumps(sources)
        response = make_response(asJson, 200)
    else:
        response = make_response("prompt parameter empty", 200)

    response.mimetype = "application/json"
    return response

@app.route("/step2", methods = ['POST'])
def step2():
    sources = list(map(lambda x: Document(page_content=x["content"], metadata=x["metadata"]), request.json))

    rPrompt = request.args.get("prompt")
    apiKey = request.args.get("openAiApiKey")
    modelName = request.args.get("modelName")
    temperature = request.args.get("temperature")
    answer = openAiCompletion(rPrompt, sources, apiKey, modelName, float(temperature)) #"text-davinci-003"
    response = make_response(answer, 200)
    response.mimetype = "text/plain"
    return response

