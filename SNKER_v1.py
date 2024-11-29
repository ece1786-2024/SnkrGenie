from transformers import pipeline
from openai import OpenAI

# Initialize sentiment analysis model
sentiment_analyzer = pipeline("text-classification", 
                            model="finiteautomata/bertweet-base-sentiment-analysis")

# Initialize OpenAI API
client = OpenAI(api_key="")

system_prompt = (
    "You are a knowledgeable sneaker salesperson. When making recommendations:"
    "1. Focus on matching the sneaker's features with the user's specific needs"
    "2. Highlight how the technical features directly benefit the user"
    "3. Do not suggest alternative products or mention limitations"
    "4. Keep the response focused on how this specific sneaker meets their needs"
    "5. Adjust your tone based on the user's tone, and make sure your response is more personable and human-like."
)

# def analyze_sentiment(user_input):
#     """Analyze the sentiment of user input"""
#     result = sentiment_analyzer(user_input)
#     sentiment = result[0]['label']  # Output 'POS', 'NEG', or 'NEU'
#     return sentiment

def analyze_sentiment(user_input):
    """Analyze the sentiment of user input"""
    try:
        # 检查输入是否为空
        if not user_input.strip():
            print("Warning: Empty input received for sentiment analysis.")
            return "NEU"  # 默认返回中性情感

        # 调用 sentiment_analyzer 进行分析
        result = sentiment_analyzer(user_input)

        # 检查返回结果是否为空
        if not result or len(result) == 0:
            print("Warning: Sentiment analyzer returned an empty result.")
            return "NEU"  # 默认返回中性情感

        # 返回分析结果
        sentiment = result[0]['label']  # 提取 'POS', 'NEG', 或 'NEU'
        return sentiment

    except IndexError as e:
        print(f"IndexError: Result index out of range for input: {user_input}")
        print(f"Error details: {e}")
        return "NEU"  # 默认返回中性情感

    except Exception as e:
        print(f"Unexpected error during sentiment analysis for input: {user_input}")
        print(f"Error details: {e}")
        return "NEU"  # 默认返回中性情感

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
