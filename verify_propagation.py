import os
import sys
import pathlib
import time
from dataclasses import dataclass, field

# Mocking parts of the environment needed for build_pipeline to run without error
sys.path.append(os.getcwd())

from dvc_viewer.parser import Pipeline, Stage, Edge, mark_stage_complete, mark_stage_failed, mark_stage_started, build_pipeline

# We want to test the propagation logic in a controlled way.
# Since build_pipeline reads from disk, we'll try to use the current trail-rag if possible,
# but for logic verification, a unit-test style approach is better.
# However, let's try to simulate the specific scenario.

project_dir = pathlib.Path('/home/lopilo/code/trail-rag')

def check_propagation():
    print("--- Propagation Test ---")
    
    # We can't easily mock the entire Pipeline object returned by parser functions 
    # because build_pipeline re-reads from disk every time.
    # Instead, let's verify the logic by calling build_pipeline and observing behavior.
    
    # 1. Initial state (everything should be consistent)
    bp = build_pipeline(project_dir)
    print(f"Initial state check for a stage with parents...")
    
    # We know in trail-rag: train_hybrid -> push_hybrid
    # If something upstream of train_hybrid is dirty, train_hybrid is dirty.
    
    # Let's use the actual mark functions.
    # We know 'split' is upstream of almost everything.
    # If we mark 'split' as failed, everything downstream should be yellow.
    mark_stage_failed('split')
    bp = build_pipeline(project_dir)
    print(f"State of 'train_hybrid' after 'split' failed: {bp.stages['train_hybrid'].state}")
    
    # Now mark 'split' as complete. 
    # train_hybrid should still be yellow if its other parents or its own status say so.
    mark_stage_complete('split')
    bp = build_pipeline(project_dir)
    print(f"State of 'train_hybrid' after 'split' complete: {bp.stages['train_hybrid'].state}")

    # Test the multi-parent scenario specifically if we can find one.
    # Looking at dvc.yaml:
    # aggregate_results depends on: benchmark_bm25s.json, benchmark_dense.json, etc.
    # source stages: benchmark_bm25s, benchmark_dense, benchmark_colbert, benchmark_hybrid, benchmark_climb
    
    print("\n--- Multi-parent Scenario (aggregate_results) ---")
    # Mark all parents as complete first to get a clean slate (if possible)
    parents = ['benchmark_bm25s', 'benchmark_dense', 'benchmark_colbert', 'benchmark_hybrid', 'benchmark_climb']
    for p in parents:
        mark_stage_complete(p)
    
    bp = build_pipeline(project_dir)
    print(f"Initial state of 'aggregate_results': {bp.stages['aggregate_results'].state}")
    
    # Mark one parent as running
    mark_stage_started('benchmark_colbert')
    bp = build_pipeline(project_dir)
    print(f"State of 'aggregate_results' when 'benchmark_colbert' is running: {bp.stages['aggregate_results'].state}")
    
    # Mark it as failed
    mark_stage_failed('benchmark_colbert')
    bp = build_pipeline(project_dir)
    print(f"State of 'aggregate_results' when 'benchmark_colbert' failed: {bp.stages['aggregate_results'].state}")
    
    # Mark it as complete, but mark ANOTHER parent as failed
    mark_stage_complete('benchmark_colbert')
    mark_stage_failed('benchmark_hybrid')
    bp = build_pipeline(project_dir)
    print(f"State of 'aggregate_results' when 'colbert' is OK but 'hybrid' failed: {bp.stages['aggregate_results'].state}")

check_propagation()
