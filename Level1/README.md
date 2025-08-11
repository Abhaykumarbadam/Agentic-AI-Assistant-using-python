# LLM-Only Smart Assistant CLI

A simple command-line assistant powered by the Groq API, utilizing the Llama3-70B model. This assistant is designed to provide clear, step-by-step answers to user queries and logs all interactions for review.

## Features

- **Step-by-Step Responses**: Delivers structured, easy-to-follow answers for complex questions.
- **Groq API Integration**: Leverages the high-speed inference capabilities of the Groq Llama3-70B model.
- **Greeting Handler**: Provides friendly, custom responses to common greetings like "hi" or "hello".
- **Math Calculation Handling**: Politely declines to perform mathematical calculations, suggesting a more appropriate tool.
- **Interaction Logging**: Automatically saves the full conversation history to `interaction_logs.json` upon exit.

## Requirements

- Python 3.8+
- `openai` Python package

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/abhaykumarbadam/llm-only-smart-assistant-cli.git
    cd llm-only-smart-assistant-cli
    ```

2.  **Install dependencies:**
    ```bash
    pip install openai
    ```

3.  **Set up your API Key:**
    Open the `chatbot.py` file and replace `"YOUR_GROQ_API_KEY_HERE"` with your actual Groq API key.
    ```python
    GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
    ```

## Usage

To start the assistant, run the following command in your terminal:

```bash
python chatbot.py
```

You will be greeted with a prompt. Type your question and press Enter.

```
GroqCloud LLM Assistant
Type your question or 'exit' to quit

You: 
```

To end the session, simply type `exit`. The conversation log will be saved to `interaction_logs.json`.

### Example Interaction

```
GroqCloud LLM Assistant
Type your question or 'exit' to quit

You: hey
Assistant: Hi, how can I help you today?
You: which is the hottest planet?
Assistant: I'd be happy to help you with that. Here are the steps to find the answer:

  Step 1: Understand the question
We want to know which planet in our solar system is the hottest.

  Step 2: Research the planets' temperatures
Let's look up the average temperatures of the planets in our solar system.

  Step 3: Compare the temperatures
After researching, we find that Venus has the highest average temperature, with surface temperatures reaching as high as 462°C (863°F).

  Step 4: Confirm the answer
Therefore, the hottest planet in our solar system is   Venus  .

There you have it!
You: exit
Session ended. Logs saved to interaction_logs.json
```

## File Structure

```
.
├── chatbot.py                 # The main application script
├── interaction_logs.json      # Auto-generated file to log conversations
└── README.md                  # This README file