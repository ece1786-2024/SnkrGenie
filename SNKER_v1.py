from transformers import pipeline
from openai import OpenAI
import os
# Initialize sentiment analysis model
sentiment_analyzer = pipeline("text-classification", 
                            model="finiteautomata/bertweet-base-sentiment-analysis")

api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI API
client = OpenAI(api_key=api_key)

system_prompt = (
    "You are a knowledgeable sneaker salesperson. When making recommendations:"
    "1. Focus on matching the sneaker's features with the user's specific needs"
    "2. Highlight how the technical features directly benefit the user"
    "3. Do not suggest alternative products or mention limitations"
    "4. Keep the response focused on how this specific sneaker meets their needs"
    "5. Adjust your tone based on the user's tone, and make sure your response is more personable and human-like."
)

def analyze_sentiment(user_input):
    """Analyze the sentiment of user input"""
    result = sentiment_analyzer(user_input)
    sentiment = result[0]['label']  # Output 'POS', 'NEG', or 'NEU'
    return sentiment

def generate_personalized_recommendation(user_input, sneaker_info, sentiment):
    """Generate personalized recommendation using GPT-4"""
    prompt = (
        f"User original input: {user_input}\n"
        f"User sentiment: {sentiment}\n"
        f"Sneaker details: Name: {sneaker_info['name']}, "
        f"Description: {sneaker_info['description']}, "
        f"Color: {sneaker_info['color']}, "
        f"Category: {sneaker_info['category']}\n"
        f"Generate a personalized recommendation for the user based on the above information."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in generating recommendation: {e}"
