import streamlit as st
from langchain.prompts import PromptTemplate
from ctransformers import AutoModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline


def complete_code(coding_lang, instruction):

    MODEL_NAME = "codellama/CodeLlama-34b-Instruct-hf"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.float16, load_in_8bit=False
    )

    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.max_new_tokens = 20
    generation_config.temperature = 0.0001
    generation_config.do_sample = True

    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        generation_config=generation_config,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

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


