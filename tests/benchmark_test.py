import manta
import time
import statistics
import psutil
import os

def get_memory_usage():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_single_analysis():
    """Run a single MANTA analysis and return execution time and memory usage"""
    file_path = "../veri_setleri/bbc_news.csv"
    column = "text"
    
    # Record initial memory
    initial_memory = get_memory_usage()
    start_time = time.time()
    
    result = manta.run_topic_analysis(
        filepath=file_path,
        column=column,
        separator=',',
        language="EN",
        tokenizer_type="bpe",
        lemmatize=True,
        generate_wordclouds=True,
        topic_count=10,
        words_per_topic=15,
        emoji_map=False,
        word_pairs_out=False,
        nmf_method="nmf"
    )
    
    end_time = time.time()
    final_memory = get_memory_usage()
    
    execution_time = end_time - start_time
    memory_used = final_memory - initial_memory
    peak_memory = final_memory
    
    return execution_time, memory_used, peak_memory, result

def main():
    print(dir(manta))
    
    # Run 10 iterations for benchmarking
    print("\n" + "="*60)
    print("MANTA PERFORMANCE BENCHMARKING - 10 ITERATIONS")
    print("="*60)
    
    execution_times = []
    memory_usages = []
    peak_memories = []
    
    print("\nRunning benchmarks...")
    for i in range(10):
        print(f"Run {i+1}/10...")
        exec_time, memory_used, peak_memory, result = run_single_analysis()
        
        execution_times.append(exec_time)
        memory_usages.append(memory_used)
        peak_memories.append(peak_memory)
        
        if result.get('state') == 'SUCCESS':
            print(f"  ✓ Completed in {exec_time:.2f}s, Memory: {memory_used:.2f} MB, Peak: {peak_memory:.2f} MB")
        else:
            print(f"  ✗ Failed: {result.get('message', 'Unknown error')}")
    
    # Calculate statistics
    print("\n" + "="*60)
    print("PERFORMANCE RESULTS")
    print("="*60)
    
    print("\nSPEED METRICS:")
    print(f"Average execution time: {statistics.mean(execution_times):.2f} seconds")
    print(f"Median execution time:  {statistics.median(execution_times):.2f} seconds")
    print(f"Min execution time:     {min(execution_times):.2f} seconds")
    print(f"Max execution time:     {max(execution_times):.2f} seconds")
    print(f"Standard deviation:     {statistics.stdev(execution_times):.2f} seconds")
    
    print("\nMEMORY USAGE METRICS:")
    print(f"Average memory used:    {statistics.mean(memory_usages):.2f} MB")
    print(f"Median memory used:     {statistics.median(memory_usages):.2f} MB")
    print(f"Min memory used:        {min(memory_usages):.2f} MB")
    print(f"Max memory used:        {max(memory_usages):.2f} MB")
    print(f"Memory usage std dev:   {statistics.stdev(memory_usages):.2f} MB")
    
    print("\nPEAK MEMORY METRICS:")
    print(f"Average peak memory:    {statistics.mean(peak_memories):.2f} MB")
    print(f"Median peak memory:     {statistics.median(peak_memories):.2f} MB")
    print(f"Min peak memory:        {min(peak_memories):.2f} MB")
    print(f"Max peak memory:        {max(peak_memories):.2f} MB")
    print(f"Peak memory std dev:    {statistics.stdev(peak_memories):.2f} MB")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Total runs: 10")
    print(f"Average time per run: {statistics.mean(execution_times):.2f}s")
    print(f"Average memory per run: {statistics.mean(memory_usages):.2f} MB")
    print(f"Total benchmark time: {sum(execution_times):.2f}s")
    print("="*60)

if __name__ == '__main__':
    main()

