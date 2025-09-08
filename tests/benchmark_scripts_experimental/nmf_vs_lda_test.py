#!/usr/bin/env python3
"""
NMF vs LDA Comparison Test

This test runs both NMF and LDA on the same dataset with identical parameters
for direct performance comparison.
"""

import os
import sys
import time
import json

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import manta
from tests.benchmark_scripts_experimental.lda_bench_test import run_lda_topic_analysis

def run_comparison_test(
    filepath: str,
    column: str,
    separator: str = ",",
    language: str = "EN", 
    topic_count: int = 5,
    words_per_topic: int = 15,
    lemmatize: bool = True
):
    """
    Run both NMF and LDA on the same dataset for comparison.
    
    Args:
        filepath: Path to data file
        column: Column containing text data
        separator: CSV separator
        language: Text language
        topic_count: Number of topics
        words_per_topic: Words per topic
        lemmatize: Whether to lemmatize
        
    Returns:
        Dictionary with comparison results
    """
    
    print("🔥 STARTING NMF VS LDA COMPARISON TEST 🔥")
    print("="*60)
    print(f"Dataset: {filepath}")
    print(f"Topics: {topic_count}")
    print(f"Language: {language}")
    print(f"Lemmatization: {lemmatize}")
    print("="*60)
    
    results = {
        "dataset": filepath,
        "parameters": {
            "topic_count": topic_count,
            "words_per_topic": words_per_topic,
            "language": language,
            "lemmatize": lemmatize
        }
    }
    
    # Run NMF Analysis
    print("\n🚀 RUNNING NMF ANALYSIS...")
    nmf_start_time = time.time()
    
    try:
        nmf_result = manta.run_topic_analysis(
            filepath=filepath,
            column=column,
            separator=separator,
            language=language,
            tokenizer_type="bpe",
            lemmatize=lemmatize,
            generate_wordclouds=True,
            topic_count=topic_count,
            words_per_topic=words_per_topic,
            emoji_map=True,
            word_pairs_out=False,
            nmf_method="nmf",
            filter_app=False,
            data_filter_options={
                "filter_app_country": "TR",
                "filter_app_country_column": "REVIEWER_LANGUAGE",
            },
            save_to_db=False
        )
        nmf_time = time.time() - nmf_start_time
        
        results["nmf"] = {
            "status": "SUCCESS",
            "execution_time": nmf_time,
            "result": nmf_result
        }
        
        print(f"✅ NMF completed in {nmf_time:.2f} seconds")
        
    except Exception as e:
        nmf_time = time.time() - nmf_start_time
        results["nmf"] = {
            "status": "FAILURE",
            "execution_time": nmf_time,
            "error": str(e)
        }
        print(f"❌ NMF failed after {nmf_time:.2f} seconds: {e}")
    
    # Run LDA Analysis  
    print("\n🚀 RUNNING LDA ANALYSIS...")
    lda_start_time = time.time()
    
    try:
        lda_result = run_lda_topic_analysis(
            filepath=filepath,
            column=column,
            separator=separator,
            language=language,
            tokenizer_type="bpe",
            lemmatize=lemmatize,
            generate_wordclouds=True,
            topic_count=topic_count,
            words_per_topic=words_per_topic,
            emoji_map=True,
            word_pairs_out=False,
            lda_method="lda",
            filter_app=False,
            data_filter_options={
                "filter_app_country": "TR",
                "filter_app_country_column": "REVIEWER_LANGUAGE",
            },
            save_to_db=False
        )
        lda_time = time.time() - lda_start_time
        
        results["lda"] = {
            "status": "SUCCESS", 
            "execution_time": lda_time,
            "result": lda_result
        }
        
        print(f"✅ LDA completed in {lda_time:.2f} seconds")
        
    except Exception as e:
        lda_time = time.time() - lda_start_time
        results["lda"] = {
            "status": "FAILURE",
            "execution_time": lda_time, 
            "error": str(e)
        }
        print(f"❌ LDA failed after {lda_time:.2f} seconds: {e}")
    
    # Generate Comparison Report
    print("\n" + "="*60)
    print("📊 COMPARISON RESULTS")
    print("="*60)
    
    if results["nmf"]["status"] == "SUCCESS" and results["lda"]["status"] == "SUCCESS":
        nmf_coherence = results["nmf"]["result"].get("coherence_score", "N/A")
        lda_coherence = results["lda"]["result"]["coherence_score"]
        
        print(f"🏆 PERFORMANCE COMPARISON:")
        print(f"   NMF Execution Time: {results['nmf']['execution_time']:.2f} seconds")
        print(f"   LDA Execution Time: {results['lda']['execution_time']:.2f} seconds")
        
        if results['nmf']['execution_time'] < results['lda']['execution_time']:
            time_diff = results['lda']['execution_time'] - results['nmf']['execution_time'] 
            print(f"   ⚡ NMF is {time_diff:.2f}s faster")
        else:
            time_diff = results['nmf']['execution_time'] - results['lda']['execution_time']
            print(f"   ⚡ LDA is {time_diff:.2f}s faster")
        
        print(f"\n🎯 QUALITY COMPARISON:")
        print(f"   NMF Coherence Score: {nmf_coherence}")
        print(f"   LDA Coherence Score: {lda_coherence:.4f}")
        
        if isinstance(nmf_coherence, (int, float)) and nmf_coherence > lda_coherence:
            print(f"   🥇 NMF has better coherence (+{nmf_coherence - lda_coherence:.4f})")
        elif isinstance(lda_coherence, (int, float)):
            if isinstance(nmf_coherence, (int, float)):
                print(f"   🥇 LDA has better coherence (+{lda_coherence - nmf_coherence:.4f})")
            else:
                print(f"   🥇 LDA coherence available, NMF coherence not available")
                
        results["comparison"] = {
            "faster_method": "NMF" if results['nmf']['execution_time'] < results['lda']['execution_time'] else "LDA",
            "time_difference": abs(results['nmf']['execution_time'] - results['lda']['execution_time']),
            "nmf_coherence": nmf_coherence,
            "lda_coherence": lda_coherence
        }
        
    else:
        print("⚠️  Cannot compare - one or both methods failed")
        if results["nmf"]["status"] == "FAILURE":
            print(f"   NMF Error: {results['nmf']['error']}")
        if results["lda"]["status"] == "FAILURE":
            print(f"   LDA Error: {results['lda']['error']}")
    
    # Save detailed results
    output_dir = os.path.join(parent_dir, "Output", "NMF_LDA_Comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, f"comparison_results_{topic_count}topics.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to: {results_file}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    # Test parameters (same as no_bench_test.py)
    file_path = "../../veri_setleri/Headlines_5000.csv"
    column = "headline"
    
    # Run comparison with different topic counts
    topic_counts = [3, 5, 7]
    
    all_results = {}
    
    for topic_count in topic_counts:
        print(f"\n{'#'*70}")
        print(f"TESTING WITH {topic_count} TOPICS") 
        print(f"{'#'*70}")
        
        result = run_comparison_test(
            filepath=file_path,
            column=column,
            separator=",",
            language="EN",
            topic_count=topic_count,
            words_per_topic=15,
            lemmatize=True
        )
        
        all_results[f"{topic_count}_topics"] = result
    
    # Final summary
    print(f"\n{'🎉'*20}")
    print("FINAL SUMMARY")
    print(f"{'🎉'*20}")
    
    for topic_count_key, result in all_results.items():
        topic_count = topic_count_key.split('_')[0]
        print(f"\n{topic_count} Topics:")
        
        if result.get("comparison"):
            comp = result["comparison"]
            print(f"  Faster: {comp['faster_method']} (by {comp['time_difference']:.2f}s)")
            print(f"  NMF Coherence: {comp['nmf_coherence']}")
            print(f"  LDA Coherence: {comp['lda_coherence']:.4f}")
        else:
            print(f"  NMF Status: {result['nmf']['status']}")
            print(f"  LDA Status: {result['lda']['status']}")
    
    print(f"\nAll comparison results saved to Output/NMF_LDA_Comparison/")