import json
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import argparse
import os
import requests

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
        steps = [line.strip() for line in response.split('\n') if line.strip()]
    return steps

def calculate_step_similarities(all_steps, model):
    """Calculate similarities between steps using sentence embeddings"""
    flat_steps = [step for problem_steps in all_steps for step in problem_steps]
    embeddings = model.encode(flat_steps)
    similarities = cosine_similarity(embeddings)
    return similarities, flat_steps

def load_hpc_instruct_data(file_path, num_samples=10):
    """Load and limit HPC-Instruct dataset"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data[:num_samples]

def main():
    parser = argparse.ArgumentParser(description='Process HPC-Instruct dataset with Ollama')
    parser.add_argument('--dataset', required=True, help='Path to HPC-Instruct dataset')
    parser.add_argument('--output_dir', required=True, help='Directory to save output')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--model_name', default='qwen', help='Name of Ollama model')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize sentence transformer for similarity analysis
    print("Loading sentence transformer model...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load dataset
    samples = load_hpc_instruct_data(args.dataset, args.num_samples)
    
    all_steps = []
    latencies = []
    
    # Process each sample
    for i, sample in enumerate(samples):
        print(f"Processing query {i+1}/{len(samples)}")
        question = sample["question"]
        prompt = f"Please solve this problem step by step:\n\n{question}"
        
        # Get response from Ollama
        response, latency = get_ollama_response(prompt, args.model_name)
        latencies.append(latency)
        
        steps = extract_steps(response)
        all_steps.append(steps)
        
        print(f"Found {len(steps)} steps. Latency: {latency:.2f} seconds")
    
    # Calculate similarities
    similarities, flat_steps = calculate_step_similarities(all_steps, st_model)
    
    # Prepare results
    results = {
        "steps": all_steps,
        "latencies": latencies,
        "flat_steps": flat_steps,
        "similarities": similarities.tolist()
    }
    
    # Save results
    output_path = os.path.join(args.output_dir, "hpc_instruct_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Processing completed and results saved to {output_path}")

if __name__ == "__main__":
    main()
