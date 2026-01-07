import time
import statistics
import subprocess
import json
import tempfile
import os


def create_worker_script(file_path="../veri_setleri/mimic_train_impressions.csv",
                         column="report",
                         topic_count=10,
                         sep=","):
    """Create a worker script that runs a single MANTA analysis"""

    # Use f-string for the entire script_content
    script_content = f'''
import manta
import resource
import platform
import json
import sys
import gc

def run_analysis():
    # Clean up memory before measurement
    gc.collect()
    
    # Run the analysis
    result = manta.run_topic_analysis(
        filepath="{file_path}",
        column="{column}",
        separator="{sep}",
        language="EN",
        lemmatize=True,
        topic_count={topic_count},
        words_per_topic=15,
        nmf_method="nmf",
        emoji_map=False,
        generate_wordclouds=True,
        save_to_db=False,
        word_pairs_out=False,
        topic_distribution=False,
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


def main(file_path, column, topic_count,sep):
    print("MANTA Performance Benchmark - Fresh Process Per Run")
    print("=" * 50)

    # Create worker script

    worker_script = create_worker_script(file_path, column, topic_count,sep)

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
                line = next(filter(lambda x: 'Gensim c_v Average:' in x, output_lines), None)
                cv_value = next((float(line.split(': ')[1]) for line in output_lines if 'Gensim c_v Average:' in line),
                                None)
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
        with open(f"benchmark_{file_path.split('/')[-1]}_{topic_count}.json", "w") as f:
            json.dump(results, f, indent=4)
        print("\nDetailed results saved to benchmark_fresh_process_results.json")
    finally:
        # Clean up temporary script
        try:
            os.unlink(worker_script)
        except OSError:
            pass


if __name__ == '__main__':
    ##("../veri_setleri/bbc_news.csv", "text"),
    ##                ("../veri_setleri/mimic_train_impressions.csv", "report")
    ## ("../veri_setleri/complaints.csv", "narrative")
    datasets = [
        #("../veri_setleri/bbc_news.csv", "text",","),
        #("../veri_setleri/complaints.csv", "narrative",";"),
        #("../veri_setleri/mimic_train_impressions.csv", "report",",")
        ("../veri_setleri/covid_abstracts.csv", "abstract",",")
    ]

    topics = [5, 10, 15]

    for file_path, column,sep in datasets:
        for topic_count in topics:
            print(f"\nRunning benchmark for topic {topic_count}...")
            print(f"Dataset: {file_path}, Column: {column}")
            column = column
            file_path = file_path
            main(file_path, column, topic_count,sep)
