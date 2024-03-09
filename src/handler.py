import runpod
from utils import JobInput
from engine import vLLMEngine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import pandas as pd

vllm_engine = vLLMEngine()

def get_dataset():
    global vectorizer,sentences,sentence_vectors,df
    if not vectorizer:
        dataset=load_dataset("meetplace1/BrainOdata",token="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")
        df = pd.DataFrame(dataset['train'])
        sentences = df["Question"].values
        vectorizer = TfidfVectorizer()
        vectorizer.fit(sentences)
        sentence_vectors = vectorizer.transform(sentences)
    return vectorizer,sentences,sentence_vectors,df

vectorizer=None
sentences=None
sentence_vectors=None
df=None

async def handler(job):
    vectorizer,sentences,sentence_vectors,df=get_dataset()
    j=job["input"]
    text=j["prompt"]
    text_vector = vectorizer.transform([text])
    similarity_scores = cosine_similarity(text_vector, sentence_vectors)
    most_similar_index = similarity_scores.argmax()
    top_indices = similarity_scores.argsort()[0][-3:]
    top=top_indices[::-1]
    top_sentences = sentences[top]
    answer = ''
    if similarity_scores[0][most_similar_index] >= 0.3:
        for sent in top_sentences:
            res = df[df["Question"] == sent]["Answer"].values[0]
            answer += res
            # txt+=res
            answer = answer.replace("•", " ")
            answer = answer.replace("\n", "")
    else:
        answer = ''    
    j["context"]=answer
    """j["sampling_params"]={
      "temperature": 0.5,
      "stop":["user","\n\n\n"],
      "top_k": -1,
      "top_p": 0.7,
      "min_p": 0.9,
      "max_tokens": 256
           }""" 
    job_input = JobInput(j)
    results_generator = vllm_engine.generate(job_input)
    async for batch in results_generator:
        yield batch
        
runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)
