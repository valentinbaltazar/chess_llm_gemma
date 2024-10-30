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
from datasets import load_dataset
import google.generativeai as genai


print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")


DESCRIPTION = """
# Chess Tutor AI
**Welcome to the Chess Chatbot!**

The goal of this project is to showcase the use of AI in learning chess. This app allows you to play a game against a custom fine-tuned model (Gemma 2B).\n
The challenge is that input must be in *algebraic notation*.

## Features

### For New & Beginner Players
- The chat interface uses the Gemini API, if you need help with chess rules or learning algebraic notation, just ask!

### For Advanced Users
- Pick an opening to play, and ask Gemini for more info.

Enjoy your game!  
**- Valentin**
"""

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key = api_key)

model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')

chat = model.start_chat()

ds = load_dataset("Lichess/chess-openings", split="train")
df = ds.to_pandas()

opening_names = df['name'].unique().tolist()


# @spaces.GPU
def generate(
    message: str,
    chat_history: list[dict],
    max_new_tokens: int = 1024,
    ) -> Iterator[str]:

    response = chat.send_message(message)

    outputs = ""
    
    for char in response:
        outputs += char
        yield outputs


def get_opening_details(opening_name):
    opening_data = df[df['name'] == opening_name].iloc[0]
    moves = opening_data['pgn']
    return f"Opening: {opening_data['name']}\nMoves: {moves}"

def get_move_list(opening_name):
    opening_data = df[df['name'] == opening_name].iloc[0]
    moves = opening_data['pgn']
    pgn_string = moves.split()
    return [move for idx,move in enumerate(pgn_string[1:],1) if idx%3!=0]
   

chat_interface = gr.ChatInterface(
    fn=generate,
    stop_btn=None,
    examples=[
        ["Hi Gemini, what is a good first move in chess?"],
        ["How does the Knight move?"],
        ["Explain algebraic notation for capturing a piece in chess?"]
    ],
    cache_examples=False,
    type="messages",
)

    
with gr.Blocks(css_paths="styles.css", fill_height=True) as demo:
    gr.Markdown(DESCRIPTION)
        
    play_match = Game()

    with gr.Row():
        with gr.Column():
            board_image = gr.HTML(play_match.display_board())
        with gr.Column():
            chat_interface.render()

    game_logs = gr.Label(label="Game Logs", elem_classes=["big-text"])
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Play a Match vs Gemma")

            move_input = gr.Textbox(label="Enter your move in algebraic notation: (e.g., e4, Nf3, Bxc4)")
            submit_move = gr.Button("Submit Move")
            submit_move.click(play_match.generate_moves, inputs=move_input, outputs=[board_image, game_logs])
            submit_move.click(lambda x: gr.update(value=''), [],[move_input])

            reset_board = gr.Button("Reset Game")
            reset_board.click(play_match.reset_board, outputs=board_image)
            reset_board.click(lambda x: gr.update(value=''), [],[game_logs])

        with gr.Column():
            gr.Markdown("### Chess Openings Explorer")

            opening_choice = gr.Dropdown(label="Choose a Chess Opening", choices=opening_names)
            opening_output = gr.Textbox(label="Opening Details", lines=4)
            opening_moves = gr.State()

            opening_choice.change(fn=get_opening_details, inputs=opening_choice, outputs=opening_output)
            opening_choice.change(fn=get_move_list, inputs=opening_choice, outputs=opening_moves)


            load_opening = gr.Button("Load Opening")
            load_opening.click(play_match.reset_board, outputs=board_image)
            load_opening.click(play_match.load_opening, inputs=[opening_choice, opening_moves], outputs=game_logs)
    
if __name__ == "__main__":
    demo.queue(max_size=20).launch()
