from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM
from langchain.vectorstores import FAISS
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from datasets import load_dataset
from pydantic import BaseModel, constr
import outlines
from flask import request, jsonify
from fastapi import FastAPI
import re
from enum import Enum
import gc
import uvicorn
from typing import List
CUDA_LAUNCH_BLOCKING=1
TF_ENABLE_ONEDNN_OPTS=0
count = 0
conversation_history = [{ "role": "user",
        "content": """You are a very sarcastic and funny farming assistant trained by cool students from Open Learning. You will answer the questions people make about farming and you will always assume the environment is a greenhouse, unless specified otherwise. Always return temperature valeus in Celcius and NEVER include fahrenheit and always try to answer the question, even if you're not sure of the answer. Your name is Botato."""},
        {"role" : "assistant", "content" : "Got it! I am a farming assistant!"}]

class Output(BaseModel):
  crop: str
  temperature: int
  humidity: int
  soil_ph: float

class ConversationEntry(BaseModel):
    role: str
    content: str

class Title(BaseModel):
    title: str

def __init__():
    global db
    global generator
    global titleGenerator
    
    dataset_name = "BotatoFontys/DataBankV2"

    loader = HuggingFaceDatasetLoader(dataset_name)

    # Load the data
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=int(512 / 10))

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

    global model

    model = outlines.models.transformers("CitrusBoy/MergedBotatoModel", device="auto", model_kwargs={ "trust_remote_code":True})

    generator = outlines.generate.json(model, Output)

    titleGenerator = outlines.generate.json(model, Title)

    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer


    global tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("CitrusBoy/FinetunedModelV2.0", padding_side="left")
    config = PeftConfig.from_pretrained("CitrusBoy/FinetunedModelV2.0")
    model = AutoModelForCausalLM.from_pretrained("CitrusBoy/FinetunedModelV2.0", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, "CitrusBoy/FinetunedModelV2.0")


__init__()

def reciprocal_rank_fusion(search_results_dict, k=60):
    fused_scores = {}
    print("Initial individual search result ranks:")
    for query, doc_scores in search_results_dict.items():
        print(f"For query '{query}': {doc_scores}")
        
    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            previous_score = fused_scores[doc]
            fused_scores[doc] += 1 / (rank + k)
            print(f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'")

    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    print("Final reranked results:", reranked_results)
    return reranked_results

app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(app)

@app.get("/inference/{crop}")
async def inference(crop : str):
    gc.collect()
    all_results = {}
    docs = []
    queries = []

    question = "What is the best mean daily temperature (celcius), humidity and ph to grow " + crop + "?"

    query = "What is the optimal mean daytime temperature (celcius) to grow " + crop + "?"
    queries.append(query)

    query = "What is the optimal mean daily humidity level to grow " + crop + "?"
    queries.append(query)

    query = "What is the optimal soil ph to grow " + crop + "?"
    queries.append(query)


    scores = {doc[0].page_content: doc[1]  for doc in docs}
    final_revision = {doc: score for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)}

    for query in queries:
        searchDocs = db.similarity_search_with_score(query, fetch_k=1, k=1)
        for doc in searchDocs:
            docs.append(doc)

        scores = {doc[0].page_content.encode().decode('unicode-escape'): doc[1]  for doc in docs}
        final_revision = {doc: score for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)}

        all_results[query] = final_revision

    reranked_results = reciprocal_rank_fusion(all_results)

    question = f"Based on these documents: {reranked_results}, answer the following question: {question}."

    print(question)

    dict = generator(question).__dict__
    print(dict)
    gc.collect()
    return jsonify(dict)
   
@app.post("/json_from_chat_history/")
def json_from_chat_history(chat_history : List[ConversationEntry]):
    gc.collect()

    formatted_history = [
        {
            "role": entry.role,
            "content": entry.content
        }
        for entry in chat_history
    ]

    print(formatted_history)

    question = f"Fill in the data based on this chat history: {formatted_history}"
    dict = generator(question).__dict__
    gc.collect()
    return dict


@app.get("/create_chat")
def create_chat():
    conversation_history = [{ "role": "user",
        "content": """You are a farming assistant trained by cool students from Open Learning. You will answer the questions people make about farming and you will always assume the environment is a greenhouse, unless specified otherwise. Always return temperature valeus in Celcius and NEVER include fahrenheit and always try to answer the question, even if you're not sure of the answer. Your name is Botato."""},
        {"role" : "assistant", "content" : "Got it! I am a farming assistant!"}]
    
    count = 0

    return {"response" : "Ok"}
    

@app.get("/chat/{message}")
async def chat(message : str):
    global count
    gc.collect()
    input = message

    conversation_history.append({"role" : "user", "content" : input})

    prompt = tokenizer.apply_chat_template(conversation_history, tokenize=False, add_generation_prompt=True)

    print(prompt)

    model_inputs = tokenizer([prompt], return_tensors="pt")

    output = model.generate(**model_inputs, max_new_tokens=500, do_sample=True, temperature=0.7)

    response = tokenizer.decode(output[0], skip_special_tokens=False)[len(prompt) - (count * 3):][:-7]

    print(response)

    conversation_history.append({"role" : "assistant", "content" : response})
    count = count + 1
    return response

@app.post("/chat/")
async def chat(chat_history : List[ConversationEntry]):
    gc.collect()

    formatted_history = [
        {
            "role": entry.role,
            "content": entry.content
        }
        for entry in chat_history
    ]

    last_entry = formatted_history[-1]

    all_results = {}
    docs = []
    queries = []

    queries.append(last_entry["content"])

    scores = {doc[0].page_content: doc[1]  for doc in docs}
    final_revision = {doc: score for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)}

    for query in queries:
        searchDocs = db.similarity_search_with_score(query, fetch_k=3, k=3)
        for doc in searchDocs:
            docs.append(doc)

        scores = {doc[0].page_content.encode().decode('unicode-escape'): doc[1]  for doc in docs}
        final_revision = {doc: score for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)}

        all_results[query] = final_revision
    
    #if there are restuls with a doc_score higher than 1, dont rerank them
    #Check if all_results.items() doc_score is higher than one. If it is, dont rerank
    if any(value < 1 for doc_score in all_results.values() for value in doc_score.values()):
        reranked_results = reciprocal_rank_fusion(all_results)
        reranked_results = {k: v for k, v in reranked_results.items() if v == max(reranked_results.values())}
        question = f"Based on these documents: {reranked_results.keys()}, answer the following question: {last_entry['content']}. If the documents have nothing to do with the question, ignore the documents completely, while still answering the question"
        formatted_history[-1]["content"] = question
        
    prompt = tokenizer.apply_chat_template(formatted_history, tokenize=False, add_generation_prompt=True)
    
    print(prompt)

    model_inputs = tokenizer([prompt], return_tensors="pt")

    output = model.generate(**model_inputs, max_new_tokens=500, do_sample=True, temperature=0.7)

    response = tokenizer.decode(output[0], skip_special_tokens=False)[len(prompt) - (len(formatted_history) - 3):][:-7]

    print(response)

    return response

@app.post("/title/")
async def title(chat_history : List[ConversationEntry]):
    gc.collect()

    formatted_history = [
        {
            "role": entry.role,
            "content": entry.content
        }
        for entry in chat_history
    ]

    prompt = tokenizer.apply_chat_template(formatted_history, tokenize=False, add_generation_prompt=True)
    
    prompt = f"Based on this chat history '{prompt}', identify the subject of the conversation and create a small title that represents it. Only return the title"


    prompt = [{"role": "user", "content" : prompt}]
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    print(prompt)

    dict = titleGenerator(prompt)

    print(dict.title)

    return dict.title
