from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import gradio as gr

mname = "NYTK/PULI-GPT-2"
#"NYTK/PULI-GPT-3SX" 
#"NYTK/PULI-GPT-2"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    device_id = torch.cuda.current_device()


tokenizer = GPT2Tokenizer.from_pretrained(mname)
model = GPT2LMHeadModel.from_pretrained(mname, pad_token_id=tokenizer.eos_token_id).to(device)

def predict(prompt):
    encoded_input = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(encoded_input, max_length=100, num_beams=15, no_repeat_ngram_size=2, early_stopping=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return response

with gr.Blocks() as demo:
    label = gr.Label('Puli GPT-2')
    input_text = gr.Textbox(label="Bemenet")
    output_text = gr.Textbox(label="Kimenet")
    predict_btn = gr.Button("Küldés")
    predict_btn.click(fn=predict, inputs=[input_text], outputs=output_text)

demo.title = "Puli GPT-2"
demo.launch()
