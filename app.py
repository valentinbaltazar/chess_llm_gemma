import os
os.environ["KERAS_BACKEND"] = "torch"  # "jax", "torch" or "tensorflow"

import gradio as gr
import keras_nlp
import keras
import spaces
import torch

from typing import Iterator
import time

from chess_board import Game

import google.generativeai as genai


print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

MAX_INPUT_TOKEN_LENGTH = 4096

MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 128

# model_id = "hf://google/gemma-2b-keras"
# model_id = "hf://google/gemma-2-2b-it"

# model_id = 'kaggle://valentinbaltazar/gemma-chess/keras/gemma_2b_en_chess'


# model = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
# tokenizer = model.preprocessor.tokenizer

DESCRIPTION = """
# Chess Tutor AI
**Welcome to the Chess Chatbot!**

The goal of this project is to showcase the use of AI in learning chess. This app allows you to play a game against a custom fine-tuned model (Gemma 2B). The challenge is that input must be in *algebraic notation*.

## Features

### For New & Beginner Players
- The chat interface uses the Gemini API, if you need help with chess rules or learning algebraic notation, just ask!

### For Advanced Users
- Pick an opening to play, and ask Gemini for more info.


<br>

Enjoy your game!  
**- Valentin**
"""

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key = api_key)

model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')

# Chat
chat = model.start_chat()

# @spaces.GPU
def generate(
    message: str,
    chat_history: list[dict],
    max_new_tokens: int = 1024,
    ) -> Iterator[str]:

    # input_ids = tokenizer.tokenize(message)
    
    # if len(input_ids) > MAX_INPUT_TOKEN_LENGTH:
    #     input_ids = input_ids[-MAX_INPUT_TOKEN_LENGTH:]
    #     gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")

    # response = model.generate(message, max_length=max_new_tokens)

    response = chat.send_message(message)

    outputs = ""
    
    for char in response.text:
        outputs += char
        yield outputs


chat_interface = gr.ChatInterface(
    fn=generate,
    stop_btn=None,
    examples=[
        ["Hi Gemma, what is a good first move in chess?"],
        ["How does the Knight move?"]
    ],
    cache_examples=False,
    type="messages",
)

    
with gr.Blocks(css_paths="styles.css", fill_height=True) as demo:
    gr.Markdown(DESCRIPTION)
        
    play_match = Game()

    # chess_png = gr.Image(play_match.display_board())
    with gr.Row():
        with gr.Column():
            board_image = gr.HTML(play_match.display_board())
        with gr.Column():
            chat_interface.render()

    game_logs = gr.Label(label="Game Logs", elem_id="game_logs_label")
    
    move_input = gr.Textbox(label="Enter your move in algebraic notation (e.g., e4, Nf3, Bxc4)")
    btn = gr.Button("Submit Move")
    btn.click(play_match.generate_moves, inputs=move_input, outputs=[board_image, game_logs])
    btn.click(lambda x: gr.update(value=''), [],[move_input])
    
    # btn.click(display_text, inputs=play_match.get_move_logs, outputs=text_output)
    

    reset_btn = gr.Button("Reset Game")
    reset_btn.click(play_match.reset_board, outputs=board_image)


if __name__ == "__main__":
    demo.queue(max_size=20).launch()
