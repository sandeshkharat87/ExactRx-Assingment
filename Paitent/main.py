from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pprint import pprint
from langchain_core.pydantic_v1 import BaseModel, Field
import gradio as gr

llm =  ChatOllama(model="cow/gemma2_tools:2b")


_template = """
You are a helpful bot. Your task is to find information from patient data. Extracts the following
properties name, gender, age, weight, height, BMI, chief medical complaint.

Always remeber:
- Dont assume informations. If only available then prase
- If you couldnt find value then value should be "not_given"

Information to understand

    "name": Name of patient , 

    "gender": Gender of patient 

    "age": Age of patient

    "weight": weight of patient,

    "height": height of patient

    "BMI": BMI of patient

    "chief_medical_complaint": chief_medical_complaint of patient


Remember:
If value not available then return value for that key should be "not_given"

patient_data: {patient_data}
data:
"""

prompt = PromptTemplate.from_template(_template)

class Patient(BaseModel):
    name: str = "Name of patient" 
    gender:str  = " Gender of patient" 
    age:str =  "Age of patient"
    weight: str =  "weight of patient,"
    height:str =  "height of patient"
    BMI :str =  "BMI of patient"
    chief_medical_complaint:str =  "chief_medical_complaint of patient"



llm_ex =  (prompt | llm.with_structured_output(Patient) )


def chat(InputData):
    response = llm_ex.invoke(InputData)
    print("%"*50)
    print(f"Input Data: {InputData}")
    pprint(f"Parsed Infor: {response}")
    print("$"*50)
    return dict(response)


UI = gr.Interface(chat,inputs="textbox", outputs="textbox")

UI.launch()

