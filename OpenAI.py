from dotenv import load_dotenv
from langchain import OpenAI
from langchain import OpenAI
from langchain import PromptTemplate
import os

load_dotenv()



# Loading the Model
openai_api_key = "API KEY"
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# Loading the data
file_path = '/Users/talib/Documents/Projects/Summary/Data/The_Fourth_AI_Inflection.txt'

with open(file_path, 'r') as file:
        essay = file.read()


# Creating a prompt template
template = """
Please write a 5 sentence summary of the following text:
{essay}
"""

prompt = PromptTemplate(
    input_variables=["essay"],
    template=template
)

summary_prompt = prompt.format(essay=essay)


# Run the model
summary = llm(summary_prompt)
print (f"Summary: {summary.strip()}")
