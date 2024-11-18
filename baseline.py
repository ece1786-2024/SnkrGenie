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

Do not include extraneous details or suggest unrelated products.
"""

def extract_and_recommend(user_input, previous_keywords=None):
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
    user_input = input("Describe your ideal sneaker: ")

    # Initial recommendation
    recommendations = extract_and_recommend(user_input)
    print("\nRecommendations:")
    print(recommendations)

    # Follow-up input
    # while True:
    #     user_input = input("\nRefine your preferences or describe further (or type 'exit' to quit): ")
    #     if user_input.lower() == "exit":
    #         break
    #     recommendations = extract_and_recommend(user_input, previous_keywords=recommendations)
    #     print("\nUpdated Recommendations:")
    #     print(recommendations)