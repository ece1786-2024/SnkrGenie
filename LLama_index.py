import pandas as pd
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from openai import OpenAI
import json
from llama_index.core.vector_stores import MetadataFilters

def initialize_index():
    """Initialize and return the index"""
    # Read CSV file
    df = pd.read_csv('t1_new205.csv')

    # Build document collection
    documents = []
    for idx, row in df.iterrows():
        text = f"""
        Name: {row['Name']}
        Description: {row['Text']}
        Colorway: {row['Colorway']}
        Main Color: {row['Main_color']}
        Category: {row['Category']}
        """
        doc = Document(text=text, metadata={
            "id": row['Id'],
            "name": row['Name'],
            "colorway": row['Colorway'],
            "main_color": row['Main_color'],
            "category": row['Category']
        })
        documents.append(doc)

    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Set global configuration
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024

    # Build and return index
    return VectorStoreIndex.from_documents(documents)

def extract_preferences_with_gpt(query, query_history):
    """Extract color and category preferences from query using GPT-4"""
    client = OpenAI(api_key="")
    
    system_prompt = """
    You are an AI assistant specialized in analyzing shoe shopping queries. 
    Please analyze the user's newest query and their query histroy and extract:
    1. Summary of the user's newest query and the query history
    2. Shoe colors they're interested in
    3. Shoe categories they're interested in (e.g., basketball shoes, running shoes)
    4. Whether the user want to end the conversation.  0 means the user want to continue. 1 means the user has got enought information and want to end the conversation.
    
    Return the results in JSON format as follows:
    {
        "query": ["summerized query"],
        "colors": ["specific color"],
        "categories": ["specific category(0 or 1)"]
        "end_flag": ["flag"]
    }
    
    If no relevant information is mentioned, the corresponding array should be empty.
    The summerized query should be based on the user's newest query and query histroy, but if there are any discrepancies, the newest query should take precedence.
    Colors and categories should be returned in English.
    The end_flag depends on user's newest query. 
    If the user indicates they have enough information or they don't need to ask anymore, that means they want to end the conversation. Otherwise they normally still want to keep asking.
    Standardize colors to: Red, Blue, Black, White, Green, Yellow, Purple, Grey.
    Standardize categories to: Basketball, Running, Lifestyle, Training.
    Example formats for "end_flag" are: "end_flag": ["1"] or "end_flag": ["0"].
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Newest query: {query}. Query history: {query_history}"}
        ],
        response_format={ "type": "json_object" }
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        return result.get('query', []), result.get('colors', []), result.get('categories', []), result.get('end_flag', [])
    except json.JSONDecodeError:
        return [], []

def get_top_matches(query: str, query_history, index: VectorStoreIndex):
    """Get best matching sneaker for the query"""
    summary, colors, categories, flag = extract_preferences_with_gpt(query, query_history)

    summary = summary[0]
    print("Summary: ", summary)
    
    filters = None
    if colors or categories:
        filter_conditions = []
        if colors:
            filter_conditions.append({
                "key": "main_color",
                "value": colors[0]
            })
        if categories:
            filter_conditions.append({
                "key": "category",
                "value": categories[0]
            })
            
        filters = MetadataFilters(filters=filter_conditions)
    
    retriever = index.as_retriever(
        similarity_top_k=1,
        filters=filters
    )
    
    retrieved_nodes = retriever.retrieve(summary)
    # retrieved_nodes = retriever.retrieve(query)
    
    # Convert to format needed by SNKER_v1.py
    top_matches = []
    for node in retrieved_nodes:
        sneaker_info = {
            "name": node.node.metadata['name'],
            "description": node.node.text,
            "color": node.node.metadata['main_color'],
            "category": node.node.metadata['category'],
            "similarity_score": node.score
        }
        top_matches.append(sneaker_info)
    
    return top_matches, summary, flag
