import time
import statistics
import subprocess
import json
import tempfile
import os
import sys

# Add parent directory to path to import lda functionality
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


def create_worker_script_lda(file_path="../veri_setleri/mimic_train_impressions.csv",
                            column="report",
                            topic_count=10):
    """Create a worker script that runs a single LDA analysis"""

    # Use f-string for the entire script_content
    script_content = f'''
import os
import sys

# Add parent directory to path to import lda functionality
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from tests.lda_bench_test import run_lda_topic_analysis
import resource
import platform
import json
import sys
import gc

def run_analysis():
    # Clean up memory before measurement
    gc.collect()
    
    # Run the LDA analysis
    result = run_lda_topic_analysis(
        filepath="{file_path}",
        column="{column}",
        separator=',',
        language="EN",
        lemmatize=True,
        generate_wordclouds=True,
        topic_count={topic_count},
        words_per_topic=15,
        emoji_map=True,
        word_pairs_out=False,
        lda_method="lda"
    )

    # Get peak memory usage throughout process lifetime
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Convert from system units to MB
    if platform.system() == 'Darwin':  # macOS
        peak_mb = peak_rss / 1024 / 1024
    else:  # Linux
        peak_mb = peak_rss / 1024

    return {{
        "peak_memory_mb": peak_mb,
    }}

if __name__ == "__main__":
    try:
        metrics = run_analysis()
        print(json.dumps(metrics))
    except Exception as e:
        print(json.dumps({{"error": str(e)}}), file=sys.stderr)
        sys.exit(1)
'''

    # Create temporary script file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        return f.name


def main(file_path, column, topic_count):
    print("LDA Performance Benchmark - Fresh Process Per Run")
    print("=" * 50)

    # Create worker script
    worker_script = create_worker_script_lda(file_path, column, topic_count)

    try:
        execution_times = []
        peak_memory_usages = []
        cv_value = None
        run_count = 10
        for i in range(run_count):
            print(f"Run {i + 1}/{run_count}...")

            start_time = time.time()

            # Run analysis in fresh Python process
            result = subprocess.run(
                ['python', worker_script],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )

            execution_time = time.time() - start_time

            if result.returncode != 0:
                print(f"  Error in run {i + 1}:")
                print(f"    STDERR: {result.stderr}")
                print(f"    STDOUT: {result.stdout}")
                continue

            # Debug: show raw output
            print(f"  Raw output: {repr(result.stdout[:200])}")
            # get coherence score from stdout
            try:
                output_lines = result.stdout.strip().split('\n')
                line = next(filter(lambda x: 'Coherence Score:' in x, output_lines), None)
                if line:
                    cv_value = float(line.split(': ')[1])
                else:
                    # Fallback: look for any coherence value in output
                    coherence_line = next(filter(lambda x: 'coherence' in x.lower() and ':' in x, output_lines), None)
                    if coherence_line and any(char.isdigit() for char in coherence_line):
                        # Try to extract float from the line
                        import re
                        numbers = re.findall(r'\d+\.?\d*', coherence_line)
                        if numbers:
                            cv_value = float(numbers[-1])  # Take the last number found

                json_line = output_lines[-1]  # Get last line which should be JSON

                metrics = json.loads(json_line)
                peak_memory_mb = metrics['peak_memory_mb']

                execution_times.append(execution_time)
                peak_memory_usages.append(peak_memory_mb)

                print(f"  Time: {execution_time:.2f}s, Peak Memory: {peak_memory_mb:.2f} MB, Coherence: {cv_value}")

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"  Failed to parse output from run {i + 1}: {e}")
                print(f"  Full stdout: {result.stdout}")
                print(f"  Full stderr: {result.stderr}")

        # Calculate standard error of the mean (SEM) for academic reporting
        import math
        sem_time = statistics.stdev(execution_times) / math.sqrt(len(execution_times)) if len(
            execution_times) > 1 else 0.0
        sem_peak = statistics.stdev(peak_memory_usages) / math.sqrt(len(peak_memory_usages)) if len(
            peak_memory_usages) > 1 else 0.0

        if execution_times:
            print("\n" + "=" * 50)
            print("RESULTS (Fresh Process Per Run)")
            print("=" * 50)

            print(f"Avg execution time: {statistics.mean(execution_times):.2f} ± {sem_time:.2f}s")
            print(f"Avg peak memory:    {statistics.mean(peak_memory_usages):.2f} ± {sem_peak:.2f} MB")
            print(f"Max peak memory:    {max(peak_memory_usages):.2f} MB")

            print("C_V Coherence value:" + (f" {cv_value:.4f}" if cv_value is not None else " N/A"))
        else:
            print("No successful runs completed.")

        # save results to a file
        if execution_times:
            results = {
                "execution_times": f"{statistics.mean(execution_times):.2f} ± {sem_time:.2f}s",
                "avg_peak_memory_mb": f"{statistics.mean(peak_memory_usages):.2f} ± {sem_peak:.2f} MB",
                "max_peak_memory_mb": f"{max(peak_memory_usages):.2f} MB",
                "cv_coherence_value": cv_value,
            }
        else:
            results = {"error": "No successful runs completed"}
        with open(f"benchmark_lda_{file_path.split('/')[-1]}_{topic_count}.json", "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nDetailed results saved to benchmark_lda_{file_path.split('/')[-1]}_{topic_count}.json")
    finally:
        # Clean up temporary script
        try:
            os.unlink(worker_script)
        except OSError:
            pass


if __name__ == '__main__':
    # Same datasets as NMF benchmark for comparison
    datasets = [
        #("../veri_setleri/bbc_news.csv", "text"),
        ("../veri_setleri/complaints.csv", "narrative"),
        #("../veri_setleri/mimic_train_impressions.csv", "report")
    ]

    topics = [5, 10, 15]

    for file_path, column in datasets:
        for topic_count in topics:
            print(f"\nRunning LDA benchmark for topic {topic_count}...")
            print(f"Dataset: {file_path}, Column: {column}")
            main(file_path, column, topic_count)