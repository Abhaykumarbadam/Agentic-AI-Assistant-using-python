import re
import json
import requests
from datetime import datetime
from calculator_tool import calculate
import os
from dotenv import load_dotenv
import nltk
from nltk import word_tokenize, pos_tag

# Uncomment once to download necessary NLTK models
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-8b-8192"

def is_math_expression(text: str) -> bool:
    return bool(re.search(r"\d+\s*[\+\-\*/]\s*\d+", text.lower()) or
                re.search(r"(add|plus|sum|subtract|minus|difference|multiply|times|product|divide|divided)", text.lower()))

def extract_math_expression(text: str) -> str:
    text = text.lower().strip()
    conversions = [
        (r'sum\s+of\s+(\d+)\s+and\s+(\d+)', r'\1 + \2'),
        (r'add\s+(\d+)\s+and\s+(\d+)', r'\1 + \2'),
        (r'add\s+(\d+)\s+to\s+(\d+)', r'\2 + \1'),
        (r'subtract\s+(\d+)\s+from\s+(\d+)', r'\2 - \1'),
        (r'(\d+)\s+minus\s+(\d+)', r'\1 - \2'),
        (r'(\d+)\s+plus\s+(\d+)', r'\1 + \2'),
        (r'multiply\s+(\d+)\s+and\s+(\d+)', r'\1 * \2'),
        (r'(\d+)\s+times\s+(\d+)', r'\1 * \2'),
        (r'divide\s+(\d+)\s+by\s+(\d+)', r'\1 / \2'),
        (r'(\d+)\s+divided\s+by\s+(\d+)', r'\1 / \2')
    ]
    for pattern, repl in conversions:
        match = re.search(pattern, text)
        if match:
            return re.sub(pattern, repl, text)
    cleaned = re.sub(r"[^\d\+\-\*/\.]", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()

def contains_question_word(text: str) -> bool:
    question_words = {"what", "who", "when", "where", "why", "how", "which", "whom", "whose"}
    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)
    return any(word in question_words and tag.startswith("W") for word, tag in tagged)

def is_greeting(text: str) -> bool:
    return text.lower() in ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]

def build_prompt(user_input: str) -> str:
    return (
        "You are a helpful assistant that always answers step-by-step.\n"
        "Avoid solving direct math calculations.\n"
        f"User question: {user_input}\n"
        "Respond clearly and logically with steps:"
    )

def call_groq_llm(prompt: str, step_by_step: bool = True) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that explains everything step-by-step." if step_by_step else
                       "You are a helpful assistant that answers concisely."
        },
        {"role": "user", "content": prompt}
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error communicating with Groq: {e}"

def split_input_into_parts(text: str):
    return re.split(r"\b(?:and also|also|then|and)\b", text, flags=re.IGNORECASE)

def chatbot():
    print("Chatbot with LLM + Calculator Tool\nType 'exit' to quit\n")
    log = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break

        entry = {
            "timestamp": str(datetime.now()),
            "user_input": user_input
        }

        responses = []
        parts = split_input_into_parts(user_input)

        for part in parts:
            part = part.strip()

            if not part:
                continue

            if is_greeting(part):
                responses.append("Hello! How can I help you today?")

            elif is_math_expression(part):
                expression = extract_math_expression(part)
                if not re.match(r"^[\d\s\+\-\*/\.]+$", expression):
                    responses.append("It looks like you're asking for a calculation and something else. Please separate them.")
                else:
                    try:
                        result = calculate(expression)
                        responses.append(f"The calculator tool is being used.\nThe result is: {result}")
                    except Exception as e:
                        responses.append(f"The calculator tool is being used.\nError in calculation: {str(e)}")

            elif contains_question_word(part):
                responses.append(call_groq_llm(part, step_by_step=False))

            else:
                prompt = build_prompt(part)
                responses.append(call_groq_llm(prompt, step_by_step=True))

        response = "\n\n".join(responses)
        print("Bot:", response)
        print()

        entry["bot_response"] = response
        log.append(entry)

    with open("interaction_logs.json", "w") as f:
        json.dump(log, f, indent=4)
        print("Interaction log saved to interaction_logs.json")

if __name__ == "__main__":
    chatbot()
