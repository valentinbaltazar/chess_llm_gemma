import os
os.environ["KERAS_BACKEND"] = "torch"  # "jax", "torch" or "tensorflow"

import gradio as gr
import keras_nlp
import keras
import spaces
import torch

from typing import Iterator
# import time

# from chess_board import Game


print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

MAX_INPUT_TOKEN_LENGTH = 4096

MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 128

model_id = "hf://google/gemma-2b-keras"


model = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
tokenizer = model.preprocessor.tokenizer

DESCRIPTION = """
# Gemma 2B
**Welcome to the Gemma Chess Chatbot!**

This game mode allows you to play a game against Gemma, the input must be in algebraic notation. \n
If you need help learning algebraic notation ask Gemma!
"""

# @spaces.GPU
def generate(
    message: str,
    chat_history: list[dict],
    max_new_tokens: int = 1024,
    ) -> Iterator[str]:

    input_ids = tokenizer.tokenize(message)
    
    if len(input_ids) > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[-MAX_INPUT_TOKEN_LENGTH:]
        gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")

    response = model.generate(message, max_length=max_new_tokens)

    outputs = ""
    
    for char in response:
        outputs += char
        yield outputs


chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
    ],
    stop_btn=None,
    examples=[
        ["Hello there! How are you doing?"],
        ["Can you explain briefly to me what is the Python programming language?"],
        ["Explain the plot of Cinderella in a sentence."],
        ["How many hours does it take a man to eat a Helicopter?"],
        ["Write a 100-word article on 'Benefits of Open-Source in AI research'"],
    ],
    cache_examples=False,
    type="messages",
)

with gr.Blocks(fill_height=True) as demo:
    gr.Markdown(DESCRIPTION)
        
    play_match = Game()

    # chess_png = gr.Image(play_match.display_board())
    with gr.Row():
        board_image = gr.HTML(play_match.display_board())
        with gr.Column():
            chat_interface.render()

    move_input = gr.Textbox(label="Enter your move in algebraic notation (e.g., e4, Nf3, Bxc4)")

    btn = gr.Button("Submit Move")
    btn.click(play_match.generate_moves, inputs=move_input, outputs=board_image)

    reset_btn = gr.Button("Reset Game")
    reset_btn.click(play_match.reset_board, outputs=board_image)


if __name__ == "__main__":
    demo.queue(max_size=20).launch()
