import httpx
import asyncio
import json
import random
from datasets import load_dataset
from datetime import datetime

API_URL = "http://localhost:8000"  
DATA_FILE = "jeebench_sample.json"

async def load_dataset_sample(sample_size=1, use_saved=True):
    """Load or download a small dataset sample from JEEBench."""
    if use_saved:
        try:
            with open(DATA_FILE, "r") as f:
                print(f"Loaded saved dataset from {DATA_FILE}")
                return json.load(f)
        except FileNotFoundError:
            print("No saved dataset found. Downloading from Hugging Face...")

    dataset = load_dataset("daman1209arora/jeebench", split="test")
    sample = random.sample(list(dataset), min(sample_size, len(dataset)))

    sample_data = [
        {
            "question": item["question"],
            "answer": item["gold"],
            "subject": item.get("subject", "")
        }
        for item in sample
    ]

    with open(DATA_FILE, "w") as f:
        json.dump(sample_data, f, indent=2)
    print(f"Saved {len(sample_data)} samples to {DATA_FILE}")

    return sample_data

async def query_agent(question: str):
    """Send question to your math agent and get response."""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(f"{API_URL}/query", json={"question": question, "user_id": "eval"})
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Error querying agent: {e}")
        return {"solution": "Error", "confidence": 0.0, "source": "error"}

def evaluate_response(response, correct_answer):
    """Check if agent's solution mentions the correct option."""
    sol = response.get("solution", "").lower()
    correct_answer = correct_answer.lower().strip()

    patterns = [
        f"answer is {correct_answer}",
        f"option {correct_answer}",
        f"({correct_answer})",
        f"{correct_answer})",
        correct_answer
    ]

    correct = any(p in sol for p in patterns)
    return correct

async def run_evaluation():
    """Main evaluation pipeline."""
    data = await load_dataset_sample(sample_size=10, use_saved=True)
    results = []

    for i, item in enumerate(data, 1):
        print(f"\n [{i}/{len(data)}] {item['subject']}")
        print("Q:", item['question'][:150], "...")
        response = await query_agent(item['question'])
        correct = evaluate_response(response, item['answer'])

        results.append({
            "question": item["question"],
            "answer": item["answer"],
            "response": response.get("solution", ""),
            "correct": correct,
            "source": response.get("source", "unknown")
        })

        print("Correct" if correct else "Incorrect")

    # Compute accuracy
    correct_count = sum(1 for r in results if r["correct"])
    acc = correct_count / len(results)
    print(f"\n Accuracy: {acc:.2%} ({correct_count}/{len(results)})")

    # Save results
    out_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
