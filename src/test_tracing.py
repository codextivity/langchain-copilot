# src/test_tracing.py
# Run this once to confirm LangSmith is receiving traces

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Simple chain — just to generate a visible trace
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("Say hello in {language}.")
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"language": "Khmer"})
print(f"Result: {result}")
print("\nNow check https://smith.langchain.com for your trace.")