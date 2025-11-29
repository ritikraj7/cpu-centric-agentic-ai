
import subprocess
from concurrent.futures import ProcessPoolExecutor

def run_command(i):
    cmd = f"python orchestrator.py {i}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(f"{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")

    return result.returncode

def main():
    # Run the command 128 times with max 64 workers
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(run_command, i) for i in range(128)]
        
        # Wait for all to complete and collect results
        results = [future.result() for future in futures]
    
    # Print summary
    successful = sum(1 for code in results if code == 0)
    failed = len(results) - successful
    print(f"Completed: {len(results)} tasks")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()