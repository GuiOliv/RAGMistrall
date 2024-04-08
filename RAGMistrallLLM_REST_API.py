from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from datasets import load_dataset
from pydantic import BaseModel, constr
import outlines
import torch
import json
from flask import Flask, request, jsonify

class Output(BaseModel):
  crop: str
  temperature: int
  humidity: int
  time_to_harvest: int
  soil_ph: float

def __init__():
    global db
    global generator

    dataset_name = "BotatoFontys/DataBank"

    loader = HuggingFaceDatasetLoader(dataset_name)

    # Load the data
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=500)

    docs = text_splitter.split_documents(data)

    modelPath = "sentence-transformers/all-MiniLM-L6-v2"

    model_kwargs = {'device':'cpu'}

    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    db = FAISS.from_documents(docs, embeddings)

    model = outlines.models.transformers("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", device="cuda")

    generator = outlines.generate.json(model, Output)


__init__()

app = Flask(__name__)

@app.get("/inference")
def inference():
    docs = []
    crop = request.args.get('crop')

    question = "What is the best mean daily temperature (celcius), humidity and ph to grow " + crop + ". How many days or years does it take to harvest a " + crop + "?"

    searchDocs = db.similarity_search("What is the best mean daily temperature (celcius) to grow " + crop + "?", fetch_k=1, k=1)
    
    for doc in searchDocs:
        docs.append(doc.page_content)

    searchDocs = db.similarity_search("What is the best mean daily humidity level to grow " + crop + "?", fetch_k=1, k=1)
    
    for doc in searchDocs:
        print(doc.page_content)
        docs.append(doc.page_content)

    searchDocs = db.similarity_search("What is the best soil ph to grow " + crop + "?", fetch_k=1, k=1)
    
    for doc in searchDocs:
        docs.append(doc.page_content)

    searchDocs = db.similarity_search("How many days or years does it take to harvest a " + crop + "?", fetch_k=1, k=1)
    
    for doc in searchDocs:
        docs.append(doc.page_content)

    joined_docs = ". ".join(docs)

    question = joined_docs + ". " + question

    print(question)

    dict = generator(question).__dict__

    return jsonify(dict)
   
