import openai
import json
import re

GROQ_API_KEY = "YOUR_GROQ_API_KEY"

if not GROQ_API_KEY or "YOUR_GROQ_API_KEY_HERE" in GROQ_API_KEY:
    raise ValueError("Please set your Groq API Key in the GROQ_API_KEY variable.")

openai.api_key = GROQ_API_KEY
openai.base_url = "https://api.groq.com/openai/v1/"
client = openai

STEP_BY_STEP_PROMPT = """You are a helpful assistant. Please always answer step by step, structuring your output in clear numbered or bulleted steps.
If asked for a math calculation (e.g., 'What is 15 + 23?'), politely refuse and suggest using a calculator.
Respond to all other questions step by step.
User question: {question}
"""

GREETING_KEYWORDS = {"hi", "hello", "hey", "hii", "heyy", "greetings"}

def strip_formatting(text):
    return re.sub(r"[*_`#>-]+", "", text).strip()

def is_greeting(text):
    return text.lower() in GREETING_KEYWORDS

def call_llm(question):
    if is_greeting(question):
        return "Hi, how can I help you today?"

    prompt = STEP_BY_STEP_PROMPT.format(question=question)
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers step-by-step."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.3,
        )
        raw_output = response.choices[0].message.content.strip()
        return strip_formatting(raw_output)
    except Exception as e:
        return f"Error: {e}"

def main():
    print("GroqCloud LLM Assistant")
    print("Type your question or 'exit' to quit\n")

    logs = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        answer = call_llm(user_input)
        print("Assistant:", answer)
        logs.append({"user": user_input, "assistant": answer})

    with open("interaction_logs.json", "w") as f:
        json.dump(logs, f, indent=2)
    print("Session ended. Logs saved to interaction_logs.json")

if __name__ == "__main__":
    main()
