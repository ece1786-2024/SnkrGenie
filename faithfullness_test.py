import pandas as pd
from datetime import datetime
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
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
    context_and_output = []  # For saving retrieval_context and actual_output
    
    for scenario, test_cases in test_cases_by_scenario.items():
        test_pairs_by_scenario[scenario] = []
        scenario_results = []
        print(f"\n=== Running tests for scenario: {scenario} ===")
        for test_case in test_cases:
            # Get first-round recommendation
            top_matches, summary, flag = get_top_matches(test_case, "", index)
            if top_matches:
                # Construct actual output and retrieval context
                actual_output = "\n".join([
                    f"Name: {match['name']}, Description: {match['description']}" for match in top_matches
                ])
                retrieval_context = [
                    f"Name: {match['name']}, Description: {match['description']}" for match in top_matches
                ]
                
                # Save retrieval_context and actual_output for each query
                context_and_output.append({
                    'scenario': scenario,
                    'query': test_case,
                    'actual_output': actual_output,
                    'retrieval_context': "\n".join(retrieval_context)
                })
                
                # Add test pair
                test_pairs_by_scenario[scenario].append((test_case, actual_output, retrieval_context))
                
                # Evaluate test case using FaithfulnessMetric and AnswerRelevancyMetric
                faithfulness_metric = FaithfulnessMetric(
                    threshold=0.7,
                    model="gpt-4",
                    include_reason=True
                )
                answerrelevancy_metric = AnswerRelevancyMetric(
                    threshold=0.7,
                    model="gpt-4",
                    include_reason=True
                )
                
                test_case_obj = LLMTestCase(
                    input=test_case,
                    actual_output=actual_output,
                    retrieval_context=retrieval_context
                )
                
                # Measure Faithfulness
                faithfulness_metric.measure(test_case_obj)
                
                # Measure Answer Relevancy
                answerrelevancy_metric.measure(test_case_obj)
                
                # Append results
                scenario_results.append({
                    'query': test_case,
                    'faithfulness_score': faithfulness_metric.score,
                    'faithfulness_reason': faithfulness_metric.reason,
                    'answer_relevancy_score': answerrelevancy_metric.score,
                    'answer_relevancy_reason': answerrelevancy_metric.reason
                })
        
        results_by_scenario[scenario] = scenario_results
    
    # Print results by scenario
    print("\n=== Test Results by Scenario ===")
    for scenario, results in results_by_scenario.items():
        total_cases = len(results)
        avg_faithfulness = sum(r['faithfulness_score'] for r in results) / total_cases if total_cases > 0 else 0
        avg_answer_relevancy = sum(r['answer_relevancy_score'] for r in results) / total_cases if total_cases > 0 else 0
        
        print(f"\n{scenario}:")
        print(f"Total cases: {total_cases}")
        print(f"Average Faithfulness Score: {avg_faithfulness:.2f}")
        print(f"Average Answer Relevancy Score: {avg_answer_relevancy:.2f}")
    
    # Summarize and save results
    detailed_results = []
    for scenario, results in results_by_scenario.items():
        for result in results:
            detailed_results.append({
                'scenario': scenario,
                **result
            })
    
    detailed_results_df = pd.DataFrame(detailed_results)
    context_and_output_df = pd.DataFrame(context_and_output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results to CSV
    detailed_results_df.to_csv(f'results_{timestamp}.csv', index=False)
    print(f"\nDetailed results saved to: results_{timestamp}.csv")
    
    # Save retrieval_context and actual_output to a separate CSV
    context_and_output_df.to_csv(f'context_output_{timestamp}.csv', index=False)
    print(f"\nContext and output saved to: context_output_{timestamp}.csv")

# Utility functions
def read_test_cases_by_scenario(filename):
    """Read test cases grouped by scenario from a file."""
    scenarios = {}
    current_scenario = None
    
    with open(filename, 'r', encoding='utf-8') as file:
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