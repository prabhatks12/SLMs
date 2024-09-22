import torch
import gradio as gr
from utils import utility
from utils import bigramModel
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

def greet(name, intensity):
    return "Hello " * intensity + name + "!"

models = ["Scratch" , "Transfer"]
bigram_model = bigramModel.model
bigram_model.load_state_dict(torch.load('working_models/bigram_model.pth'))


neo_model = GPTNeoForCausalLM.from_pretrained("gpt_neo/")
neo_tokenizer = GPT2Tokenizer.from_pretrained("gpt_neo/")


def generate_text(model, text, max_length):
    max_length = int(max_length)
    if model == "Scratch":
        text = text.lower()
        print("text ", text)
        enc = utility.encode(text) # dogs often referred to as
        print(f"BigramModel: Encoded : {enc}")
        decode_ls = utility.decode(bigram_model.generate(torch.tensor([enc]), max_new_tokens=max_length)[0].tolist())
        print(f"BigramModel: Decoded : {decode_ls}")
        return decode_ls
    else:
        input_ids = neo_tokenizer(text, return_tensors="pt")
        input_ids["attention_mask"] = (input_ids["input_ids"] != neo_tokenizer.pad_token_id).long()
        print(f"GPT_NEO: Encoded : {input_ids}")
        outputs = neo_model.generate(input_ids.input_ids, max_length=max_length, pad_token_id=neo_tokenizer.pad_token_id)
        print(f"GPT_NEO: Decoded : {outputs}")
        return neo_tokenizer.decode(outputs[0])


demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Dropdown(models, label="Model"),
        gr.Textbox(label="Text"),
        gr.Slider(1, 30, value=10, step=1, label="Select the number of words"),
    ],
    outputs=gr.Label(label="Generate"),
    title="Text Generation",
    description="Select a model and enter text to generate more",
)

demo.launch()