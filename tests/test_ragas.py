import pytest
import sys
import os
import json
from unittest.mock import patch

# Добавляем корневую директорию в sys.path, чтобы импортировать модуль
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_ragas_demo_test import main

def test_ragas_pipeline():
    """
    Integration test for Ragas pipeline.
    Runs the main function which:
    1. Loads goldens
    2. Generates answers using LLM
    3. Evaluates using Ragas
    4. Checks thresholds
    """
    # Use a small subset for testing or mock LLM if execution is slow/expensive
    # checking if tests/goldens.json exists
    if not os.path.exists("tests/goldens.json"):
        pytest.skip("tests/goldens.json not found")
        
    # We can invoke main with arguments
    # Ensure output goes to a test-specific file
    output_file = "tests/test_results.json"
    
    # Run the main function
    # It raises SystemExit(1) on failure, SystemExit(0) or returns None on success
    test_args = ["--goldens", "tests/goldens.json", "--output", output_file]
    
    try:
        main(test_args)
    except SystemExit as e:
        # Check exit code
        if e.code is not None and e.code != 0:
             pytest.fail(f"Ragas pipeline failed with exit code {e.code}")
        
    # Verify results file exists and has content
    assert os.path.exists(output_file), "Results file was not created"
    with open(output_file, 'r') as f:
        results = json.load(f)
        assert len(results) > 0, "Results file is empty"
        # Check if metrics are present in the first result
        first = results[0]
        # At least one metric should be there (e.g. faithfulness if context present)
        # Note: 'qa_semantic_correctness' is always computed in the script
        assert "qa_semantic_correctness" in first or "answer_relevancy" in first, "Metrics missing in results"
