# pip install transformers datasets evaluate --quiet
#!pip install transformers datasets evaluate rouge_score
from transformers import pipeline
import evaluate
summarizer = pipeline("summarization", model="t5-small")
text = """Artificial Intelligence is transforming industries by automating tasks,
improving decision-making, and creating new opportunities for innovation."""
summary = summarizer(text, max_length=30, min_length=5, do_sample=False)[0]['summary_text']
rouge = evaluate.load("rouge")
results = rouge.compute(predictions=[summary], references=[text])
print("Original Text:\n", text)
print("\nGenerated Summary:\n", summary)
print("\nROUGE Evaluation:\n",results)



# !pip install sentence-transformers
# !pip install faiss-cpu
documents = [
    "The Earth revolves around the Sun.",
    "The Moon affects the tides of the Earth.",
    "Albert Einstein developed the theory of relativity.",
    "Python is a programming language used for machine learning.",
    "Photosynthesis is the process by which plants make food using sunlight."
]
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoTokenizer,AutoModelForCausalLM
embedder=SentenceTransformer("all-MiniLM-L6-v2")
doc_embedding=embedder.encode(documents)
dimension=doc_embedding.shape[1]
index=faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embedding))
def retrive(query,k=2):
    query_embedding=embedder.encode([query])
    D,I =index.search(query_embedding,k)
    retrived_doc=[documents[i] for i in I[0]]
    return retrived_doc
tokenizer=AutoTokenizer.from_pretrained('distilgpt2')
model=AutoModelForCausalLM.from_pretrained('distilgpt2')
def generate(query):
    context_doc=retrive(query,k=2)
    context=" ".join(context_doc)
    promt=f"context:{context} \n Question:{query}\n ans:"
    inputs=tokenizer(promt,return_tensors='pt',truncation=True)
    output=model.generate(inputs["input_ids"],max_new_tokens=50)
    answer =tokenizer.decode(output[0],skip_special_tokens=True)
    return answer
query="what cause the tides?"
response=generate(query)
print(response) 

import torch 
from transformers import DistilBertModel , DistilBertTokenizer
import time
import psutil
model_name="distilbert-base-uncased"
tokenizer=DistilBertTokenizer.from_pretrained(model_name)
model=DistilBertModel.from_pretrained(model_name)
text="genai is optimized using quantization or pruning"
inputs=tokenizer(text,return_tensors='pt')
start_time=time.time()
original_output=model(**inputs)
end_time=time.time()
original_time = end_time - start_time
original_memory = psutil.Process().memory_info().rss / (1024 ** 2)
print(f"Before Quantization - Inference Time: {original_time:.4f}s, Memory: {original_memory:.2f} MB")
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
start_time = time.time()
quant_output = quantized_model(**inputs)
end_time = time.time()
quant_time = end_time - start_time
quant_memory = psutil.Process().memory_info().rss / (1024 ** 2)
print(f"After Quantization - Inference Time: {quant_time:.4f}s, Memory: {quant_memory:.2f} MB")
# === Step 7: Compare Performance ===
print("\n=== Evaluation ===")
print(f"Speed Improvement: {((original_time - quant_time) / original_time) * 100:.2f}% faster")
print(f"Memory Reduction: {(original_memory - quant_memory):.2f} MB less used")



