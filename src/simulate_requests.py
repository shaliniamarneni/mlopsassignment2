import os
import requests
import random
import time
from pathlib import Path

API_URL = "http://localhost:5000/predict"
TEST_DATA_DIR = Path("data/processed/test")

def simulate():
    if not TEST_DATA_DIR.exists():
        print(f"Test data directory not found at {TEST_DATA_DIR}. Run preprocessing.")
        return

    classes = ["Cat", "Dog"]
    test_files = []
    
    for cls in classes:
        cls_dir = TEST_DATA_DIR / cls
        if cls_dir.exists():
            for f in os.listdir(cls_dir):
                if f.endswith(".jpg"):
                    test_files.append((cls_dir / f, cls))
                    
    if not test_files:
        print("No test files found.")
        return
        
    # Pick a random sample
    sample_files = random.sample(test_files, min(10, len(test_files)))
    
    print(f"--- Simulating traffic for {len(sample_files)} images ---")
    correct = 0
    total_time = 0
    
    for file_path, true_label in sample_files:
        start_time = time.time()
        
        try:
            with open(file_path, "rb") as f:
                response = requests.post(API_URL, files={"file": f})
                
            latency = time.time() - start_time
            total_time += latency
            
            if response.status_code == 200:
                result = response.json()
                pred_label = result["predicted_class"]
                conf = result["confidence"]
                
                is_correct = pred_label == true_label
                if is_correct:
                    correct += 1
                    
                match_str = "MATCH" if is_correct else "MISMATCH"
                print(f"[{match_str}] True: {true_label} | Pred: {pred_label} (Conf: {conf:.2f}) | Latency: {latency:.4f}s")
            else:
                print(f"API Error {response.status_code} for {file_path.name}")
                
        except Exception as e:
            print(f"Request failed: {e}")
            
    print(f"--- Simulation Complete ---")
    print(f"Accuracy: {correct}/{len(sample_files)} ({correct/len(sample_files)*100:.1f}%)")
    print(f"Average Latency: {total_time/len(sample_files):.4f}s")

if __name__ == "__main__":
    simulate()
