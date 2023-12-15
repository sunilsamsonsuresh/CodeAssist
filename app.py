import streamlit as st
from huggingface_hub import hf_hub_download
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from ctransformers import AutoModelForCausalLM


def complete_code(coding_lang, instruction):
    llm = CTransformers(model='/content/drive/MyDrive/Colab/model/codellama-13b-instruct.Q4_K_M.gguf',
                      model_type='llama',
                      config={'max_new_tokens':512,
                              'temperature':0.01,
                              'gpu_layers': 50})

    template = """
    You are an exceptionally intelligent {coding_lang} coding assistant that \n
    consistently delivers accurate and reliable responses to user instructions.

    You are supposed to output only the code based on the instruction given below.

    {instruction}

    """

    prompt = PromptTemplate(input_variables=["coding_lang", "instruction"],
                            template=template)

    response = llm(prompt.format(coding_lang=coding_lang, instruction=instruction))

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
    code_start_index = output_text.find("*/")

    text_before_code = output_text[:code_start_index]
    code_portion = output_text[code_start_index + 2:]

    st.text(text_before_code)

    # Display the code in a separate box
    st.code(code_portion)