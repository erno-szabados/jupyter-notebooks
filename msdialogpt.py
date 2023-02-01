from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch
#HF_HUB_DISABLE_SYMLINKS_WARNING=1

mname = 'microsoft/DialoGPT-large'
#mname = 'microsoft/DialoGPT-small'
tokenizer = AutoTokenizer.from_pretrained(mname, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(mname)

def predict(input, history=[]):
    # tokenize the new input sentence
    new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([torch.LongTensor(history), new_user_input_ids], dim=-1)

    # generate a response
    history = model.generate(
        bot_input_ids,
        max_length=1000,
        num_beams = 5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        do_sample=True,
        top_p=0.92,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id).tolist()

    # convert the tokens to text, and then split the responses into lines
    response = tokenizer.decode(history[0]).split("<|endoftext|>")
    response = [(response[i], response[i+1]) for i in range(0, len(response)-1, 2)]  # convert to tuples of list
    return response, history

demo = gr.Interface(
    title='DialoGPT',
    theme='grass',
    fn=predict,
    inputs = ["text", "state"],
    outputs = ['chatbot', "state"],
    allow_flagging='never'
).launch()
