import pandas as pd
from datetime import datetime
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from LLama_index import initialize_index, get_top_matches

# Main function
def main():
    # Initialize index
    index = initialize_index()
    
    # Read test cases by scenario
    test_cases_by_scenario = read_test_cases_by_scenario('input_test.txt')
    
    # Organize test pairs by scenario
    test_pairs_by_scenario = {}
    results_by_scenario = {}
    
    for scenario, test_cases in test_cases_by_scenario.items():
        test_pairs_by_scenario[scenario] = []
        scenario_results = []
        print(f"\n=== Running tests for scenario: {scenario} ===")
        for test_case in test_cases:
            # Get first-round recommendation
            top_matches, summary, flag = get_top_matches(test_case, "", index)
            print(index)
            if top_matches:
                # Construct actual output and retrieval context
                actual_output = "\n".join([
                    f"Name: {match['name']}, Description: {match['description']}" for match in top_matches
                ])
                retrieval_context = [
                    f"Name: {match['name']}, Description: {match['description']}" for match in top_matches
                ]
                
                # Add test pair
                test_pairs_by_scenario[scenario].append((test_case, actual_output, retrieval_context))
                
                # Evaluate test case using FaithfulnessMetric
                faithfulness_metric = FaithfulnessMetric(
                    threshold=0.7,
                    model="gpt-4",
                    include_reason=True
                )
                test_case_obj = LLMTestCase(
                    input=test_case,
                    actual_output=actual_output,
                    retrieval_context=retrieval_context
                )
                faithfulness_metric.measure(test_case_obj)
                
                scenario_results.append({
                    'query': test_case,
                    'faithfulness_score': faithfulness_metric.score,
                    'reason': faithfulness_metric.reason
                })
        
        results_by_scenario[scenario] = scenario_results
    
    # Print results by scenario
    print("\n=== Test Results by Scenario ===")
    for scenario, results in results_by_scenario.items():
        total_cases = len(results)
        avg_faithfulness = sum(r['faithfulness_score'] for r in results) / total_cases if total_cases > 0 else 0
        
        print(f"\n{scenario}:")
        print(f"Total cases: {total_cases}")
        print(f"Average Faithfulness Score: {avg_faithfulness:.2f}")
    
    # Summarize and save results
    detailed_results = []
    for scenario, results in results_by_scenario.items():
        for result in results:
            detailed_results.append({
                'scenario': scenario,
                **result
            })
    
    detailed_results_df = pd.DataFrame(detailed_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_results_df.to_csv(f'faithfulness_results_{timestamp}.csv', index=False)
    
    print(f"\nDetailed results saved to: faithfulness_results_{timestamp}.csv")

# Utility functions
def read_test_cases_by_scenario(filename):
    """Read test cases grouped by scenario from a file."""
    scenarios = {}
    current_scenario = None
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                current_scenario = line[1:].strip()
                scenarios[current_scenario] = []
            elif current_scenario:
                scenarios[current_scenario].append(line)
    
    return scenarios

# Run main
if __name__ == "__main__":
    main()