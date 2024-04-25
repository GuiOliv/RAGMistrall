from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from datasets import load_dataset
from pydantic import BaseModel, constr
import outlines
from flask import Flask, request, jsonify
from enum import Enum
import gc
CUDA_LAUNCH_BLOCKING=1
TF_ENABLE_ONEDNN_OPTS=0
class Environment(str, Enum):
    Outdoors ="Outdoors"
    Greenhouse = "Greenhouse"
    Dont_know = "Dont know"

class Lightning(str, Enum):
    Direct_sunlight = "Direct Sunlight"
    Shade = "Shade"
    Dont_know = "Dont know"

class Output(BaseModel):
  crop: str
  temperature: int
  humidity: int
  soil_ph: float
  environment: Environment
  lightning: Lightning

def __init__():
    global db
    global generator

    dataset_name = "BotatoFontys/DataBankV2"

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

    model = outlines.models.transformers("meta-llama/Llama-2-7b-chat-hf", device="auto")

    generator = outlines.generate.json(model, Output)


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

app = Flask(__name__)

@app.get("/inference")
def inference():
    all_results = {}
    docs = []
    queries = []
    crop = request.args.get('crop')

    question = "What is the best mean daily temperature (celcius), humidity and ph to grow " + crop + ". What is the best environment to grow the crop in?"

    query = "What is the optimal mean daytime temperature (celcius) to grow " + crop + "?"
    queries.append(query)

    query = "What is the optimal mean daily humidity level to grow " + crop + "?"
    queries.append(query)

    query = "What is the optimal soil ph to grow " + crop + "?"
    queries.append(query)

    query = "What is the best environment to grow " + crop + "? Outdoors or Greenhouse?"
    queries.append(query)

    query = "What is the best lightning condition to grow " + crop + "? Direct sunlight or Shade?"
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

    question = f"Based on these documents: {reranked_results}, answer the following question: {question}. If you don't know the answer, return that you dont know."

    print(question)

    dict = generator(question).__dict__
    print(dict)
    gc.collect()
    return jsonify(dict)
   
