# Agentic-AI-with-Multi-Step-Tasks

This repository contains a Python-based command-line chatbot that demonstrates agentic AI principles. The agent can understand and execute multi-step user prompts by breaking them down into individual tasks. It intelligently selects and uses appropriate tools—such as a calculator for math and a translator for language—or falls back to a large language model for general knowledge queries. All interactions are processed using the fast Groq API and logged for review.

## Key Features

*   **Multi-Step Task Decomposition:** Parses complex commands like *"Translate 'Good Morning' into German and then multiply 5 and 6"* into sequential actions.
*   **Tool-Based Function Execution:** Utilizes specialized tools for specific tasks:
    *   `calculator_tool.py` for mathematical calculations.
    *   `translator_tool.py` for English to German translation.
*   **LLM Fallback:** For general questions or tasks that don't match a specific tool, it queries the Groq Llama3-8B model for a factual answer.
*   **Interaction Logging:** Saves a complete record of user inputs, individual step processing, and final bot responses to `interactions_logs.json`.
*   **Groq API Integration:** Leverages the Groq API for high-speed LLM inference.

## How It Works

The core logic in `full_agent.py` drives the agent's behavior. When a user provides input, the agent:

1.  **Splits the input** into distinct steps based on keywords like `then` or punctuation.
2.  For each step, it **uses regular expressions** to identify specific intents (e.g., `translate '...' into german`, `add X and Y`).
3.  If a known intent is matched, it **invokes the corresponding Python function** from the toolset (`translate_to_german()` or `calculate()`).
4.  If no specific tool is matched, the step is treated as a general query and **sent to the Groq LLM**.
5.  All responses from the individual steps are **aggregated and delivered** to the user as a single, coherent answer.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Abhaykumarbadam/Agentic-AI-with-Multi-Step-Tasks.git
    cd Agentic-AI-with-Multi-Step-Tasks
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Key:**
    Create a file named `.env` in the root directory and add your Groq API key:
    ```
    GROQ_API_KEY='your_api_key_here'
    ```

## Usage

Run the chatbot from your terminal:

```bash
python full_agent.py
```

The agent will prompt you for input. Type your requests and press Enter. To quit the application, type `exit`.

### Example Interactions

```
Full Agentic AI
Type 'exit' to quit

You: Translate 'Good Morning' into German and then multiply 5 and 6.
Bot: Translation result: Guten Morgen
The calculator tool is being used.
The result is: 30.0

You: Tell me the capital of Italy, then add 12 and 12.
Bot: The capital of Italy is Rome.
The calculator tool is being used.
The result is: 24.0

You: What is the distance between Earth and Mars?
Bot: The average distance between Earth and Mars is about 225 million kilometers (140 million miles). This distance varies greatly due to their elliptical orbits around the Sun.

You: exit
Bot: Goodbye!
Saved to interaction_logs.json