# SnkrGenie

## System Overview
SnkrGenie is a multi-round chatbot designed to recommend sneakers that match users' specific needs and preferences

## File Descriptions

### main.py
This file contains the main application logic for the Sneaker Recommendation Chatbot. It includes the GUI setup using Tkinter, methods for processing user input, and recommendation functions.

### t1_finished.csv
This CSV file contains a dataset of sneaker information, including details such as name, colorway, main color, category, retail price, and brand. It is used to build the document collection for the recommendation system.

### g_evaluation_summary_20241124_192608.csv
This file provides a summary of evaluation results for different recommendation scenarios, including metrics like relevance, completeness, personalization, expertise, and language.

### g_evaluation_detailed_20241124_192608.csv
This file contains detailed evaluation results for each test case, including specific strengths, weaknesses, and suggestions for improvement.

### performance_test.ipynb
A Jupyter Notebook used for performance testing of the recommendation system. It includes code for initializing the system, running test cases, and evaluating results.

### LLama_index.py
This file contains functions for initializing the index, extracting preferences using GPT-4, and retrieving top sneaker matches based on user queries.

### SNKER_v1.py
This file includes functions for sentiment analysis and generating personalized sneaker recommendations using OpenAI's GPT-4 model.

### input_test.txt
A text file containing test cases grouped by different scenarios, such as activity, style, comfort, occasion, and brand impression, used for evaluating the recommendation system.

### faithfullness_test.py
This script is used to evaluate the faithfulness of the recommendation system's outputs against the retrieval context, using metrics like faithfulness and answer relevancy.

### faithfullness_results.csv
This CSV file contains the results of the faithfulness evaluation, including scores and reasons for each test case.

## Core Features
1. User Input Processing
2. Intent & Preference Analysis
3. Product Retrieval
4. Personalized Recommendations
5. Continuous Conversation

## Architecture

### Main Program (main.py)
- GUI Implementation
- Dialogue Management
- Component Coordination

### Retrieval Engine (LLama_index.py)
- LlamaIndex Vector Index Construction
- User Preference & Intent Extraction
- Similarity-based Product Matching

### Recommendation Generator (SNKER_v1.py)
- Personalized Message Generation

## Usage Guide
### Running the System
1. Install the required packages for each Python and notebook file, see imports in each file.
2. Fill in the API keys for each file.
3. Launch main program: `python main.py`
   - Starts the GUI interface
4. Enter your vague description about sneakers in the GUI.
5. End Conversation: Enter information about wanting to end the conversation.

### Running Evaluations

#### Faithfulness Test
Run `faithfulness_test.py` to generate:
- `faithfulness_results.csv`: Detailed faithfulness scores
- `context_output_{timestamp}.csv`: Context-output pairs

#### Performance Test
Run `performance_test.ipynb` to generate:
- `g_evaluation_detailed_{timestamp}.csv`: Detailed evaluation scores
- `g_evaluation_summary_{timestamp}.csv`: Evaluation metrics summary
