import json
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import argparse
import os
import requests
from datasets import load_dataset  # Import Hugging Face datasets library

def get_ollama_response(prompt, model_name="qwen"):
    """Get response from Ollama API"""
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    start_time = time.time()
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        end_time = time.time()
        
        result = response.json()
        return result['response'], end_time - start_time
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return "", 0

def extract_steps(response):
    """Extract numbered steps from the response"""
    steps = re.findall(r'\d+\.\s+(.*?)(?=\n\d+\.|\n*$)', response, re.DOTALL)
    
    if not steps:
        steps = [line.strip() for line in response.split('\n') 
                if line.strip() and not line.startswith("Here") and not line.startswith("Let")]
    
    if not steps:
        steps = [response.strip()]
        
    return steps

def calculate_step_similarities(all_steps, model):
    """Calculate similarities between steps using sentence embeddings"""
    flat_steps = [step for problem_steps in all_steps for step in problem_steps]
    embeddings = model.encode(flat_steps)
    similarities = cosine_similarity(embeddings)
    return similarities, flat_steps

def load_first_10_questions():
    """Load the first 10 questions from the HPC-Instruct dataset using Hugging Face datasets library"""
    dataset = load_dataset("hpcgroup/hpc-instruct", split="train[:10]")  # Load first 10 samples
    questions = [{"question": item["problem_statement"]} for item in dataset]
    print(f"Successfully loaded first 10 questions from the HPC-Instruct dataset")
    return questions

def main():
    parser = argparse.ArgumentParser(description='Process first 10 questions from HPC-Instruct dataset with Ollama')
    parser.add_argument('--output_dir', required=True, help='Directory to save output')
    parser.add_argument('--model_name', default='qwen', help='Name of Ollama model')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading sentence transformer model...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    samples = load_first_10_questions()
    
    all_steps = []
    latencies = []
    responses = []
    
    for i, sample in enumerate(samples):
        print(f"\nProcessing question {i+1}/10")
        question = sample["question"]
        print(f"Question: {question[:100]}...")
        
        prompt = f"Please solve this problem step by step:\n\n{question}"
        
        response, latency = get_ollama_response(prompt, args.model_name)
        latencies.append(latency)
        responses.append(response)
        
        steps = extract_steps(response)
        all_steps.append(steps)
        
        print(f"Found {len(steps)} steps. Latency: {latency:.2f} seconds")
    
    similarities, flat_steps = calculate_step_similarities(all_steps, st_model)
    
    results = {
        "metadata": {
            "total_questions": 10,
            "model_name": args.model_name,
            "average_latency": sum(latencies) / len(latencies)
        },
        "questions": [sample["question"] for sample in samples],
        "full_responses": responses,
        "extracted_steps": all_steps,
        "latencies": latencies,
        "flat_steps": flat_steps,
        "similarities": similarities.tolist()
    }
    
    output_path = os.path.join(args.output_dir, "hpc_instruct_first_10_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessing completed and results saved to {output_path}")
    print(f"Average latency: {results['metadata']['average_latency']:.2f} seconds")

if __name__ == "__main__":
    main()
