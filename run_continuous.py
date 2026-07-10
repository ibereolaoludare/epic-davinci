import os
import time
import subprocess

print("Starting continuous Train-Backtest loop...")
iteration = 1

while True:
    print(f"\n======================================")
    print(f"      LOOP ITERATION {iteration}")
    print(f"======================================")
    
    print("-> 1. Training AI (1 Million Steps)...")
    try:
        subprocess.run(["python", "train_model.py"], check=True)
    except subprocess.CalledProcessError:
        print("Training failed. Retrying in 10 seconds...")
        time.sleep(10)
        continue
    
    print("-> 2. Running Backtest...")
    try:
        subprocess.run(["python", "backtest.py"], check=True)
    except subprocess.CalledProcessError:
        print("Backtest failed. Continuing...")
    
    print("-> 3. Backtest complete. Saving iteration snapshot...")
    # Optionally save the backtest chart for this iteration
    if os.path.exists('backtest_results.png'):
        snapshot_name = f'backtest_results_iter_{iteration}.png'
        # Windows copy command via shell
        subprocess.run(f'copy backtest_results.png {snapshot_name}', shell=True)
        print(f"Saved {snapshot_name}")
    
    print("Waiting 5 seconds before next iteration...")
    time.sleep(5)
    iteration += 1
