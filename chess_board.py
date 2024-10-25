# import os
# os.environ["KERAS_BACKEND"] = "torch"  # "jax", "torch" or "tensorflow"

import keras_nlp
import keras
import torch

import chess
import chess.svg
import time

class Game:
    def __init__(self):
        # Initialize the chess board
        self.board = chess.Board()
        self.sequence = []
        self.counter = 0
        self.model_id = 'kaggle://valentinbaltazar/gemma-chess/keras/gemma_2b_en_chess'
        self.sampler = keras_nlp.samplers.TopKSampler(k=50, temperature=0.7)
        self.model = keras_nlp.models.GemmaCausalLM.from_preset(self.model_id)
        self.compile_model()

    def compile_model(self):
        self.model.compile(sampler=self.sampler)
    
    def call_gemma(self):
        template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
        

        prompt = template.format(
            instruction=f"Predict the next chess move in the sequence {str(self.sequence)}",
            response="",)

        output = self.model.generate(prompt, max_length=256)
       
        gemma_move = output.split(' ')[-1].strip("'")

        # gemma_move = 'e5'

        if self.make_move(gemma_move):
            print(f'Gemma plays {self.sequence[-1]}! (Current Sequence: {self.sequence} {len(self.sequence)})')
            self.counter = 0
            return self.display_board()
        elif self.counter < 10:
            self.counter += 1
            print(self.counter)
            self.call_gemma()
        else:
            print("Gemma quit...")
            return None

    def gemma_moves(self):
        print(f"Gemma is thinking...(Current Sequence: {self.sequence} {len(self.sequence)})")
        time.sleep(3)
        return self.call_gemma()

    def player_moves(self, move):
        return self.make_move(move)

    # Function to display the board
    def display_board(self):
        # clear_output(wait=True)
        # display(SVG(chess.svg.board(board=self.board)))
        board_svg = chess.svg.board(board=self.board)
        # return svg2png(bytestring=board_svg)
        return board_svg

    # Function to make a move
    def make_move(self, move):
        try:
            update = self.board.parse_san(move)
            self.board.push(update)
            # self.display_board()
            self.sequence.append(move)
            return self.display_board()
        except:
            print(f"Invalid move '{move}'. Use algebraic notation (e.g., 'e4', 'Nf3', 'Bxc4') or ask Gemma for help.")
            return None
        
    def reset_board(self):
        self.board = chess.Board()
        self.sequence = []
        self.counter = 0
        # self.board.reset
        return self.display_board()
    
    def generate_moves(self, move):
        yield self.player_moves(move)
        yield self.gemma_moves()

def main():
    end_game = False # Change this to False

    play_match = Game()
    play_match.display_board()

    while end_game is False:
        move = input("Your move (or 'No' to end game):")
        if 'No' in move:
            del play_match
            end_game = True
        else:
            play_match.player_moves(move)

if __name__ == '__main__':
    main()