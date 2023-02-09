#-----------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
#-----------------------------------------------------------------------------------------

# import torch
from flask import Flask, make_response, request
#from langchain_bot import *
import urllib.parse
import requests
import re
from bs4 import BeautifulSoup

from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document

def constructUrl(path):
    clientId = "c196d990-f964-457f-5690-552e7a52600c"
    baseUrl = "https://acehelphub.humany.net/ace-help-hub-86/"
    site = "%2F%2Facehelphub.humany.net%2Fadmin%2Finterfaces"
    funnel = "ace-help-hub-86"
    return baseUrl + path + "?client=" + clientId + "&funnel=" + funnel + "&site=" + site

def constructQueryUrl(phrase):
    return constructUrl("guides") + "&phrase=" + urllib.parse.quote(phrase) + "&skip=0&take=6&sorting.type=popularity&sorting.direction=descending"
def constructGuideUrl(guideId):
    return constructUrl("guides/" + guideId)

def getArticle(id):
    url = constructGuideUrl(id)
    resp = requests.post(url).json()
    # body = resp["Body"]
    body = BeautifulSoup(resp["Body"], "html.parser").get_text() #html5lib
    # sources.append(resp["Title"] + " \n" + body)
    return { "title": resp["Title"], "page_content": body, "metadata": {"source": url} }

def getRelevantSources(phrase):
    x = requests.post(constructQueryUrl(phrase))

    totalLen = 0
    sources = []
    for match in x.json()["Matches"]:
        print(match["Title"])
        relativeUrl = match["RelativeUrl"]
        rx = re.search("^\d+", relativeUrl) 
        id = rx.group(0)
        
        article = getArticle(id)
        content = article["page_content"]
        totalLen += len(content)
        if totalLen > 4000 / 1.4:
            break
        else:
            sources.append(Document(
                page_content=content,
                metadata=article["metadata"]
            )
    )
    return sources

def createAnswer(question):
    sources = getRelevantSources(question)
    chain = load_qa_with_sources_chain(
        OpenAI(temperature=0, openai_api_key="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")
        # model_name="text-davinci-003" https://platform.openai.com/docs/models
        # max_tokens=256 # sets an upper bound on how many tokens the API will return
    ) #, chain_type="map_reduce")
    
    return chain(
        {
            "input_documents": sources,
            "question": question,
        },
        return_only_outputs=True,
    )["output_text"]


app = Flask(__name__)

@app.route("/")
def hello():
    rPrompt = request.args.get("prompt")
    if not(rPrompt is None) and len(rPrompt):
        answer = createAnswer(rPrompt)
        response = make_response(answer, 200)
    else:
        response = make_response("prompt parameter empty", 200)

    response.mimetype = "text/plain"
    return response
