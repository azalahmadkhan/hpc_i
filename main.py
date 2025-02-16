import json
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_model_response(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )
    end_time = time.time()
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    latency = end_time - start_time
    return response, latency

def extract_steps(response):
    steps = re.findall(r'\d+\.\s+(.*?)(?=\n\d+\.|\n*$)', response, re.DOTALL)
    if not steps:
        steps = [line.strip() for line in response.split('\n') if line.strip()]
    return steps

def calculate_step_similarities(all_steps, model):
    flat_steps = [step for problem_steps in all_steps for step in problem_steps]
    embeddings = model.encode(flat_steps)
    similarities = cosine_similarity(embeddings)
    return similarities, flat_steps

def load_hpc_instruct_data(file_path, num_samples=10):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data[:num_samples]

def main():
    parser = argparse.ArgumentParser(description='Process HPC-Instruct dataset with Transformers')
    parser.add_argument('--dataset', required=True, help='Path to HPC-Instruct dataset')
    parser.add_argument('--output_dir', required=True, help='Directory to save output')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--model_path', default='Qwen/Qwen-7B-Chat', help='Path or name of model')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    print(f"Loading model {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Initialize sentence transformer
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    samples = load_hpc_instruct_data(args.dataset, args.num_samples)
    
    all_steps = []
    latencies = []
    
    for i, sample in enumerate(samples):
        print(f"Processing query {i+1}/{len(samples)}")
        question = sample["question"]
        prompt = f"Please solve this problem step by step:\n\n{question}"
        
        response, latency = get_model_response(prompt, model, tokenizer, device)
        latencies.append(latency)
        
        steps = extract_steps(response)
        all_steps.append(steps)
        
        print(f"Found {len(steps)} steps. Latency: {latency:.2f} seconds")
        
    similarities, flat_steps = calculate_step_similarities(all_steps, st_model)
    
    results = {
        "steps": all_steps,
        "latencies": latencies,
        "flat_steps": flat_steps,
        "similarities": similarities.tolist()
    }
    
    output_path = os.path.join(args.output_dir, "hpc_instruct_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Processing completed and results saved to {output_path}")

if __name__ == "__main__":
    main()
