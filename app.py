import streamlit as st
from huggingface_hub import hf_hub_download
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from ctransformers import AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


model_name = "Deci/DeciLM-7B-instruct"

device = "cuda" # for GPU usage or "cpu" for CPU usage

bnb_config = BitsAndBytesConfig(
    load_in_8bit_fp32_cpu_offload = True,
    #load_in_8bit = True,
    bnb_8bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


def complete_code(coding_lang, instruction):
    llm = pipeline("text-generation",
                          model=model,
                          tokenizer=tokenizer,
                          temperature=0.1,
                          device_map="auto",
                          max_length=4096,
                          return_full_text=False)

    template ="""
    ### System:
    You are an exceptionally intelligent programmer who \n
    consistently delivers accurate and reliable responses to user instructions.
    ### User:
    {instruction} in {coding_lang}
    ### Assistant:
    """

    prompt = PromptTemplate(input_variables=["coding_lang", "instruction"],
                            template=template)

    response = llm(prompt.format(coding_lang=coding_lang, instruction=instruction))
    response = response[0]['generated_text']
    return response


st.set_page_config(
    page_title="CodeAssist",
    page_icon='</>',
    layout='centered',
    initial_sidebar_state='collapsed')

st.header('Coding Assistant')

coding_lang = st.selectbox('Choose the language',
                          ('Python', 'C++', 'Java', 'JavaScript'), index=0)

instruction = st.text_area("Your input here !")

submit = st.button("Generate")

if submit:
    output_text = complete_code(coding_lang, instruction)
    code_start_index = output_text.find("```")

    text_before_code = output_text[:code_start_index]
    code_portion = output_text[code_start_index + 2:]

    st.text(text_before_code)

    # Display the code in a separate box
    st.code(code_portion)