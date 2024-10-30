import os
os.environ["KERAS_BACKEND"] = "torch"  # "jax", "torch" or "tensorflow"

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
        self.arrow= None
        
        self.model_id = 'kaggle://valentinbaltazar/gemma-chess/keras/gemma_2b_en_chess'
        self.sampler = keras_nlp.samplers.TopKSampler(k=50, temperature=0.7)
        self.model = keras_nlp.models.GemmaCausalLM.from_preset(self.model_id)
        self.compile_model()

    def compile_model(self):
        self.model.compile(sampler=self.sampler)
    
    def call_gemma(self, opening_move):
        template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
        
        if opening_move:
            gemma_move = opening_move
        else:
            template = "Instruction:\n{instruction}\n\nResponse:\n{response}"

            prompt = template.format(
                instruction=f"Predict the next chess move in the sequence {str(self.sequence)}",
                response="",)

            output = self.model.generate(prompt, max_length=256)
        
            gemma_move = output.split(' ')[-1].strip("'")

        if self.make_move(gemma_move):
            print(f'Gemma plays {self.sequence[-1]}! (Current Sequence: {self.sequence} {len(self.sequence)})')
            self.counter = 0
            return self.display_board()
        elif self.counter < 10:
            self.counter += 1
            print(self.counter)
            return self.call_gemma()
        else:
            print("Gemma quit...")
            return None

    def gemma_moves(self):
        """Calls Gemma to make a move, either self generated or from opening sequence"""
        if self.opening_moves and len(self.sequence)<len(self.opening_moves):
            return self.call_gemma(self.opening_moves[len(self.sequence)])
        else:
            return self.call_gemma(None)

    def player_moves(self, move):
        return self.make_move(move)

    def display_board(self):
        """Return SVG image of board state"""
        if self.arrow:
            board_svg = chess.svg.board(board=self.board, arrows=[self.arrow])
        else:
            board_svg = chess.svg.board(board=self.board)
        return board_svg

   
    def make_move(self, move):
        """Checks to see if move is valid, if so pushes move to board state"""
        try:
            update = self.board.parse_san(move)
            self.board.push(update)
            self.sequence.append(move)
            self.arrow = chess.svg.Arrow(update.from_square, update.to_square, color="#0000cccc")
            return True
        except:
            print(f"Invalid move '{move}'. Use algebraic notation (e.g., 'e4', 'Nf3', 'Bxc4') or ask Gemma for help.")
            return False
        
    def reset_board(self):
        self.board = chess.Board()
        self.sequence = []
        self.counter = 0
        self.arrow = None
        return self.display_board()
    
    def generate_moves(self, move):
        """Generator function for one full turn of chess moves"""
        valid_move = self.player_moves(move)
        if valid_move:
            yield self.display_board(), f"You played: {move}"
            time.sleep(2)
            yield self.display_board(), f"Gemma is thinking...(Current Sequence: {self.sequence} {len(self.sequence)})"
            time.sleep(3)
            yield self.gemma_moves(), f'Gemma plays {self.sequence[-1]}! (Current Sequence: {self.sequence} {len(self.sequence)})'
        else:
            print("Try again")
            yield self.display_board(), "Try again"

    def get_move_logs(self):
        return self.sequence
    
    def load_opening(self, opening_name, opening_moves):
        self.opening = True
        self.opening_name = opening_name
        self.opening_moves = opening_moves
        return f"Ok, lets play the {opening_name}! {opening_moves} Make your first move."
        

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