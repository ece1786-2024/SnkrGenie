from openai import OpenAI
import os

# Initialize OpenAI API
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# System prompt for GPT-4
system_prompt = """
You are a knowledgeable sneaker expert. Based on a user's input, you will:
1. Analyze the user's preferences and dislikes regarding sneakers.
2. Recommend sneakers that match their description.
3. Avoid suggesting sneakers that don't align with their dislikes.
4. Provide a brief explanation for why each sneaker fits their needs.
"""

def generate_customer_request(sneaker_description):
    """
    Call GPT-4 API to generate a single sentence requirement or scenario for a given sneaker description.
    """
    prompt = (
    f"Imagine I am a customer considering this shoe: {sneaker_description}. "
    "Based on its features, especial the technical features"
    "write a paragraph that captures what I might say to a salesperson "
    "and don't mention the shoe's name. Make it sound like a real customer request. Make it ambiguous."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        customer_sentence =  response.choices[0].message.content.strip()
        return customer_sentence
    except Exception as e:
        return f"Error in generating customer sentence: {e}"

def extract_and_recommend(user_input):
    # Construct prompt
    baseline_prompt = (
        f"User input: {user_input}\n"
        f"Generate a personalized recommendation for the user based on these options, "
        f"highlighting how each sneaker meets their specific needs."
    )

    try:
        # Call GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": baseline_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        recommendation = response.choices[0].message.content.strip()
        return recommendation
    except Exception as e:
        return f"Error in generating recommendation: {e}"

if __name__ == "__main__":
    # Get user input
    user_input = input("Describe sneaker: ")
    simulated_user = generate_customer_request(user_input)
    print("\nSimulated User:")
    print(simulated_user)
    print("--------------------")
    # recommendations = extract_and_recommend(simulated_user)
    # print("\nRecommendations:")
    # print(recommendations)
