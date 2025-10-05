import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# 1) Load environment variales and API key
load_dotenv("apikey.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# 2) Simple calculator tool (OK for homework/demo only)
def calculator(expression: str) -> str:
    try:
        result = eval(expression)  # ⚠️ don't use eval on untrusted input
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error: {e}"

system_msg = "You are a helpful assistant with access to a calculator function."
question = "What is 12 * (7 + 3) - 96?"

# 3) Ask LLM; declare tool
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }]
)

msg = response.choices[0].message

if msg.tool_calls:
    tc = msg.tool_calls[0]
    # 4) arguments are JSON string -> parse them
    args = tc.function.arguments
    if isinstance(args, str):
        args = json.loads(args)

    expression = args.get("expression", "")
    tool_result = calculator(expression)

    # 5) Send tool result back to LLM
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
            msg,  # the tool call message
            {"role": "tool", "tool_call_id": tc.id, "content": tool_result}
        ]
    )

# 6) Print final answer
print("LLM Final Answer:", response.choices[0].message.content)
