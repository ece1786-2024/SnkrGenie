import sys
import json
from openai import OpenAI
from LLama_index import initialize_index, get_top_matches
from SNKER_v1 import analyze_sentiment, generate_personalized_recommendation
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime
import os
# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def create_customer_agent():
    """Create customeragent prompt"""
    system_prompt = """
    You are a customer looking for sneakers. Your task is to evaluate if the salesperson's recommendation fully addresses your initial requirements.

    Evaluation rules:
    1. Carefully compare the recommendation against your initial requirements
    2. Consider the recommendation satisfactory ONLY IF it addresses ALL aspects of your initial requirements
    3. If ALL requirements are met:
       - Set "satisfied" to true
       - Set "end_conversation" to true
       - Express satisfaction in your response
    4. If ANY requirement is not addressed:
       - Set "satisfied" to false
       - Set "end_conversation" to false
       - In your response, clearly state which requirements were not met
    5. Maximum 3 conversation turns

    Return response in JSON format:
    {
        "satisfied": true/false,
        "response": "your response",
        "end_conversation": true/false,
        "requirements_met": ["list of met requirements"],
        "requirements_missing": ["list of unmet requirements"]
    }

    Example:
    If initial request was "comfortable for walking and stylish", and recommendation only addresses comfort,
    you should identify that style was not addressed and request more information about style aspects.
    """
    return system_prompt

def evaluate_conversation(initial_query, recommendation):
    """Evaluate the conversation"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": create_customer_agent()},
                {"role": "user", "content": f"Initial requirement: {initial_query}\nSalesperson recommendation: {recommendation}"}
            ],
            response_format={ "type": "json_object" }
        )
        
        # Parse the response
        evaluation = json.loads(response.choices[0].message.content)
        
        # # Print the conversation details
        # print("\n=== Conversation Turn ===")
        # print(f"Customer's initial request: {initial_query}")
        # print(f"Salesperson's recommendation: {recommendation}")
        # print(f"Customer's response: {evaluation['response']}")
        # print(f"Customer satisfied: {'Yes' if evaluation['satisfied'] else 'No'}")
        # print(f"End conversation: {'Yes' if evaluation['end_conversation'] else 'No'}")
        # print("=====================\n")
        
        return evaluation
    except Exception as e:
        print(f"Evaluation error: {e}")
        return None
    
def read_test_cases_by_scenario(filename: str) -> Dict[str, List[str]]:
    """Read test cases from file and organize them by scenario"""
    scenarios = {}
    current_scenario = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                # New scenario found
                current_scenario = line.replace('#', '').strip()
                scenarios[current_scenario] = []
            elif current_scenario and line[0].isdigit():
                # Remove the numbering (e.g., "1. ") from the line
                test_case = line[line.find(' ')+1:].strip()
                scenarios[current_scenario].append(test_case)
    
    return scenarios

def run_test_case(query, index):
    """Run a single test case"""
    conversation_history = []
    max_turns = 3
    current_turn = 0
    current_query = query
    
    while current_turn < max_turns:
        # Get recommendations
        top_matches, summary, flag = get_top_matches(current_query, " ".join(conversation_history), index)
        if not top_matches:
            return False, current_turn + 1
            
        # Generate recommendation response
        sentiment = analyze_sentiment(current_query)
        recommendation = generate_personalized_recommendation(summary, top_matches[0], sentiment)
        
        # Evaluate recommendation
        evaluation = evaluate_conversation(query, recommendation)
        if evaluation is None:
            return False, current_turn + 1
            
        conversation_history.append(current_query)
        
        if evaluation['satisfied'] or evaluation['end_conversation']:
            return True, current_turn + 1
            
        current_query = evaluation['response']
        current_turn += 1
    
    return False, max_turns


def create_evaluator_agent():
    """Create the evaluator system prompt"""
    system_prompt = """
    You are a professional recommendation system evaluator. Please evaluate the system's responses 
    across the following dimensions:
    
    1. Relevance: How well the recommendation matches user needs (1-5 points)
    2. Completeness: Whether all user requirements are addressed (1-5 points)
    3. Personalization: Level of recommendation personalization (1-5 points)
    4. Expertise: Level of product knowledge demonstrated (1-5 points)
    5. Language: Clarity and naturalness of expression (1-5 points)
    
    Return evaluation in JSON format:
    {
        "scores": {
            "relevance": float,
            "completeness": float,
            "personalization": float,
            "expertise": float,
            "language": float,
            "average": float
        },
        "analysis": {
            "strengths": ["strength1", "strength2"...],
            "weaknesses": ["weakness1", "weakness2"...],
            "suggestions": ["suggestion1", "suggestion2"...]
        }
    }
    """
    return system_prompt

def g_evaluate(query: str, recommendation: str) -> Dict:
    """Perform G-Evaluation assessment"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": create_evaluator_agent()},
                {"role": "user", "content": f"User Query: {query}\nSystem Recommendation: {recommendation}"}
            ],
            response_format={ "type": "json_object" }
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"G-Evaluation error: {e}")
        return None

def run_evaluation_batch(test_cases_by_scenario: Dict[str, List[Tuple[str, str]]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run batch G-Evaluation and generate reports by scenario"""
    all_results = []
    scenario_summaries = []
    
    for scenario, cases in test_cases_by_scenario.items():
        scenario_results = []
        
        for query, recommendation in cases:
            evaluation = g_evaluate(query, recommendation)
            if evaluation:
                result = {
                    'scenario': scenario,
                    'query': query,
                    'recommendation': recommendation,
                    **evaluation['scores'],
                    'strengths': '; '.join(evaluation['analysis']['strengths']),
                    'weaknesses': '; '.join(evaluation['analysis']['weaknesses']),
                    'suggestions': '; '.join(evaluation['analysis']['suggestions'])
                }
                scenario_results.append(result)
                all_results.append(result)
        
        # Calculate scenario averages
        if scenario_results:
            df_scenario = pd.DataFrame(scenario_results)
            scenario_summary = {
                'Scenario': scenario,
                'Average Relevance': df_scenario['relevance'].mean(),
                'Average Completeness': df_scenario['completeness'].mean(),
                'Average Personalization': df_scenario['personalization'].mean(),
                'Average Expertise': df_scenario['expertise'].mean(),
                'Average Language': df_scenario['language'].mean(),
                'Overall Average': df_scenario['average'].mean()
            }
            scenario_summaries.append(scenario_summary)
    
    return pd.DataFrame(all_results), pd.DataFrame(scenario_summaries)

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
        
        for test_case in test_cases:
            # Get first-round recommendation
            top_matches, summary, flag = get_top_matches(test_case, "", index)
            if top_matches:
                sentiment = analyze_sentiment(test_case)
                recommendation = generate_personalized_recommendation(summary, top_matches[0], sentiment)
                test_pairs_by_scenario[scenario].append((test_case, recommendation))
                
                success, turns = run_test_case(test_case, index)
                scenario_results.append({
                    'query': test_case,
                    'success': success,
                    'turns': turns
                })
        
        results_by_scenario[scenario] = scenario_results
    
    # Print results by scenario
    print("\n=== Test Results by Scenario ===")
    for scenario, results in results_by_scenario.items():
        total_cases = len(results)
        successful_cases = sum(1 for r in results if r['success'])
        success_rate = (successful_cases / total_cases) * 100 if total_cases > 0 else 0
        avg_turns = sum(r['turns'] for r in results) / total_cases if total_cases > 0 else 0
        
        print(f"\n{scenario}:")
        print(f"Total cases: {total_cases}")
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Average turns: {avg_turns:.2f}")
    
    # Run G-Evaluation
    print("\n=== G-Evaluation Results ===")
    detailed_results, scenario_summary = run_evaluation_batch(test_pairs_by_scenario)
    
    # Display scenario summary table
    print("\nScenario Summary:")
    print(scenario_summary.to_string(index=False))
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_results.to_csv(f'g_evaluation_detailed_{timestamp}.csv', index=False)
    scenario_summary.to_csv(f'g_evaluation_summary_{timestamp}.csv', index=False)
    
    print(f"\nDetailed results saved to: g_evaluation_detailed_{timestamp}.csv")
    print(f"Summary results saved to: g_evaluation_summary_{timestamp}.csv")

if __name__ == "__main__":
    main()