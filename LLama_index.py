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

def extract_preferences_with_gpt(query):
    """Extract color and category preferences from query using GPT-4"""
    client = OpenAI(api_key="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")
    
    system_prompt = """
    You are an AI assistant specialized in analyzing shoe shopping queries. 
    Please analyze the user's query and extract:
    1. Shoe colors they're interested in
    2. Shoe categories they're interested in (e.g., basketball shoes, running shoes)
    
    Return the results in JSON format as follows:
    {
        "colors": ["specific color"],
        "categories": ["specific category"]
    }
    
    If no relevant information is mentioned, the corresponding array should be empty.
    Colors and categories should be returned in English.
    Standardize colors to: Red, Blue, Black, White, Green, Yellow, Purple, Grey
    Standardize categories to: Basketball, Running, Lifestyle, Training
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        response_format={ "type": "json_object" }
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        return result.get('colors', []), result.get('categories', [])
    except json.JSONDecodeError:
        return [], []

def get_top_matches(query: str, index: VectorStoreIndex):
    """Get best matching sneaker for the query"""
    colors, categories = extract_preferences_with_gpt(query)
    
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
    
    retrieved_nodes = retriever.retrieve(query)
    
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
    
    return top_matches
