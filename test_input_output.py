import pandas as pd
from datetime import datetime
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from LLama_index import initialize_index, get_top_matches

# Main function
def main():
    # Load the dataset
    dataset_path = "shoe_recommendations.csv"  # Replace with your dataset path
    data = pd.read_csv(dataset_path)

    # Initialize index
    index = initialize_index()

    # Store results
    results = []
    context_and_output = []  # For saving retrieval_context and actual_output

    print("\n=== Running tests for dataset ===")
    for _, row in data.iterrows():
        question = row['Question']
        expected_answer = row['Answer']
        
        # Generate model's response
        top_matches, summary, flag = get_top_matches(question, "", index)
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
                'question': question,
                'actual_output': actual_output,
                'retrieval_context': "\n".join(retrieval_context)
            })
            
            # Evaluate using FaithfulnessMetric and AnswerRelevancyMetric
            faithfulness_metric = FaithfulnessMetric(
                threshold=0.7,
                model="gpt-4o-mini",  # Use the faster, cheaper model
                include_reason=True
            )
            answerrelevancy_metric = AnswerRelevancyMetric(
                threshold=0.7,
                model="gpt-4o-mini",  # Use the faster, cheaper model
                include_reason=True
            )
            
            test_case_obj = LLMTestCase(
                input=question,
                actual_output=actual_output,
                expected_output=expected_answer
            )
            
            # Measure Faithfulness
            faithfulness_metric.measure(test_case_obj)
            
            # Measure Answer Relevancy
            answerrelevancy_metric.measure(test_case_obj)
            
            # Append results
            results.append({
                'question': question,
                'expected_answer': expected_answer,
                'actual_output': actual_output,
                'faithfulness_score': faithfulness_metric.score,
                'faithfulness_reason': faithfulness_metric.reason,
                'answer_relevancy_score': answerrelevancy_metric.score,
                'answer_relevancy_reason': answerrelevancy_metric.reason
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    context_and_output_df = pd.DataFrame(context_and_output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results to CSV
    results_file = f'results_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Save retrieval_context and actual_output to a separate CSV
    context_file = f'context_output_{timestamp}.csv'
    context_and_output_df.to_csv(context_file, index=False)
    print(f"\nContext and output saved to: {context_file}")

if __name__ == "__main__":
    main()