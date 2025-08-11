import re
import json
import requests
from datetime import datetime
from calculator_tool import calculate
from translator_tool import translate_to_german
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-8b-8192"

def call_groq_llm(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant that responds clearly and accurately."},
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
        return f"[LLM error: {str(e)}]"


def is_greeting(text):
    return text.lower().strip() in ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]

def extract_all_math_expressions(text):
    conversions = [
        (r'add (\d+)\s+(and|to)\s+(\d+)', lambda m: f"{m.group(1)} + {m.group(3)}"),
        (r'sum of (\d+)\s+and\s+(\d+)', lambda m: f"{m.group(1)} + {m.group(2)}"),
        (r'multiply (\d+)\s+and\s+(\d+)', lambda m: f"{m.group(1)} * {m.group(2)}"),
        (r'(\d+)\s+times\s+(\d+)', lambda m: f"{m.group(1)} * {m.group(2)}"),
        (r'(\d+)\s*\+\s*(\d+)', lambda m: f"{m.group(1)} + {m.group(2)}"),
        (r'(\d+)\s*\*\s*(\d+)', lambda m: f"{m.group(1)} * {m.group(2)}")
    ]
    text = text.lower()
    expressions = []

    for pattern, builder in conversions:
        for match in re.finditer(pattern, text):
            expr = builder(match)
            expressions.append(expr)
            text = text.replace(match.group(0), "", 1)

    return expressions

def extract_translations(text):
    match1 = re.search(r"translate\s+'(.+?)'\s+into\s+german", text, re.IGNORECASE)
    match2 = re.search(r"(.+?)\s+in\s+german", text, re.IGNORECASE)
    if match1:
        return match1.group(1)
    elif match2:
        return match2.group(1)
    return None

def process_step(step):
    step = step.strip()

    if is_greeting(step):
        return "Hello! How can I assist you today?"

    # Translation
    phrase = extract_translations(step)
    if phrase:
        result = translate_to_german(phrase)
        return f"Translation result: {result}"

    # Math
    math_exprs = extract_all_math_expressions(step)
    if math_exprs:
        responses = []
        for expr in math_exprs:
            try:
                result = calculate(expr)
                responses.append(f"The calculator tool is being used.\nThe result is: {result}")
            except Exception as e:
                responses.append(f"The calculator tool is being used.\nError in calculation: {str(e)}")
        return "\n".join(responses)

    # LLM fallback
    return call_groq_llm(step)


def chatbot():
    print("Full Agentic AI\nType 'exit' to quit\n")
    log = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break

        entry = {
            "timestamp": str(datetime.now()),
            "user_input": user_input,
            "steps": []
        }

        steps = re.split(r"\bthen\b|\.\s*", user_input, flags=re.IGNORECASE)
        steps = [s.strip() for s in steps if s.strip()]

        responses = []
        for step in steps:
            response = process_step(step)
            responses.append(response)
            entry["steps"].append({"step": step, "response": response})

        final_response = "\n".join(responses)
        print("Bot:", final_response, "\n")

        entry["bot_response"] = final_response
        log.append(entry)

    with open("interaction_logs.json", "w") as f:
        json.dump(log, f, indent=4)
        print("Saved to interaction_logs.json")

if __name__ == "__main__":
    chatbot()
