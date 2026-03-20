import subprocess
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_training(timesteps, lr, ent_coef, gamma, suffix):
    cmd = [
        ".\\venv\\Scripts\\python.exe", "train_rl_real_data.py",
        "--timesteps", str(timesteps),
        "--lr", str(lr),
        "--ent-coef", str(ent_coef),
        "--gamma", str(gamma)
    ]
    logging.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    try:
        with open("models/training_meta.json", "r") as f:
            meta = json.load(f)
            # Evaluate using forward out_of_sample or test data
            fwd_metrics = meta.get("out_of_sample", {}).get("fwd", {})
            test_metrics = meta.get("out_of_sample", {}).get("test", {})
            
            # Using forward data metrics if available, otherwise test
            metrics = fwd_metrics if fwd_metrics else test_metrics
            
            if metrics:
                profit = metrics.get("strategy_return_pct", -999)
                sharpe = metrics.get("sharpe_ratio", -999)
                return profit, sharpe, metrics
            return -999, -999, {}
    except Exception as e:
        logging.error(f"Failed to read meta: {e}")
        return -999, -999, {}

def main():
    logging.info("Starting Auto-Tuner...")
    
    param_grid = [
        {"lr": 3e-4, "ent_coef": 0.05, "gamma": 0.99},
        {"lr": 1e-4, "ent_coef": 0.01, "gamma": 0.999},
        {"lr": 5e-4, "ent_coef": 0.08, "gamma": 0.95},
    ]
    
    best_sharpe = -999
    best_params = None
    
    for idx, params in enumerate(param_grid):
        logging.info(f"--- Testing Config {idx+1}/{len(param_grid)}: {params} ---")
        # Run a short training to test parameters
        profit, sharpe, metrics = run_training(
            timesteps=150_000, 
            lr=params["lr"], 
            ent_coef=params["ent_coef"], 
            gamma=params["gamma"], 
            suffix=str(idx)
        )
        
        logging.info(f"Result: Profit={profit}%, Sharpe={sharpe}, Details={metrics}")
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params
            
    logging.info(f"==========================================")
    logging.info(f"Best Config Found: {best_params} with Sharpe={best_sharpe}")
    logging.info(f"Starting FULL training with best params...")
    logging.info(f"==========================================")
    
    if best_params:
        subprocess.run([
            ".\\venv\\Scripts\\python.exe", "train_rl_real_data.py",
            "--timesteps", "2500000",
            "--lr", str(best_params["lr"]),
            "--ent-coef", str(best_params["ent_coef"]),
            "--gamma", str(best_params["gamma"])
        ])
        logging.info("Full training completed!")

if __name__ == "__main__":
    main()
