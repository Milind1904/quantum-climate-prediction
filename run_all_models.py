import subprocess
import sys
import time

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*70}")
    print(f"[RUNNING] {description}")
    print(f"{'='*70}")
    
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error in {script_name}: {e}")
        return False

def main():
    print(f"{'='*70}")
    print("QUANTUM vs CLASSICAL CLIMATE PREDICTION")
    print("Complete Pipeline Execution")
    print(f"{'='*70}")
    
    scripts = [
        ('classical_lstm_model.py', 'Classical LSTM Model Training'),
        ('quantum_qsvc_model.py', 'Quantum QSVC Model Training'),
        ('compare_models.py', 'Model Comparison & Analysis')
    ]
    
    total_start = time.time()
    completed = 0
    
    for script, description in scripts:
        if run_script(script, description):
            completed += 1
        else:
            print(f"Skipping remaining scripts due to error.")
            break
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*70}")
    print("PIPELINE EXECUTION SUMMARY")
    print(f"{'='*70}")
    print(f"Scripts completed: {completed}/{len(scripts)}")
    print(f"Total execution time: {total_time/60:.1f} minutes")
    
    if completed == len(scripts):
        print(f"\n✓ ALL MODELS TRAINED AND COMPARED SUCCESSFULLY!")
        print(f"\nGenerated files:")
        print(f"  ✓ classical_lstm_results.png")
        print(f"  ✓ quantum_qsvc_results.png")
        print(f"  ✓ model_comparison_complete.png")
        print(f"  ✓ classical_results.json")
        print(f"  ✓ quantum_results.json")
        print(f"  ✓ summary_report.txt")
    else:
        print(f"\n⚠ Pipeline incomplete. Check errors above.")

if __name__ == "__main__":
    main()