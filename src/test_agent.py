# src/test_agent.py

from dotenv import load_dotenv
load_dotenv()

from agent import build_research_agent, run_agent

agent = build_research_agent()

print("=" * 60)
print("TEST 1: Pure calculation")
print("=" * 60)
answer = run_agent(
    agent,
    "Cambodia's GDP was $15,228M in 2013 and $31,940M in 2023. "
    "Calculate the exact CAGR over 10 years."
)
print(answer)

print("\n" + "=" * 60)
print("TEST 2: Date reasoning")
print("=" * 60)
answer = run_agent(
    agent,
    "How many years ago was 2013?"
)
print(answer)

print("\n" + "=" * 60)
print("TEST 3: Multi-step reasoning")
print("=" * 60)
answer = run_agent(
    agent,
    "If Cambodia maintains its average growth rate of 9.43% annually, "
    "what would its GDP be in 5 years from now? Start from $31,940M."
)
print(answer)