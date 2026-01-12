---
title: Chess with Gemma LLM
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
short_description: AI chess chatbot with a custom fine-tuned Gemma model
---

# Chess Tutor AI: Fine-Tuned LLM for Chess Move Prediction

**[Try the Live Demo](https://huggingface.co/spaces/valentin-ub/chess_llm_gemma)**

An interactive chess application powered by a custom fine-tuned Gemma 2B model that predicts chess moves. This project explores whether large language models can learn strategic game play through fine-tuning on chess game data.

## Project Overview

This project investigates a key question in AI: **Can LLMs learn to play chess?** Rather than using traditional chess engines with minimax algorithms, I fine-tuned Google's Gemma 2B model to predict chess moves by learning patterns from thousands of chess games.

### What I Built

- **Fine-tuned Gemma 2B** on chess game sequences using instruction-tuning format
- **Inference pipeline** that generates moves given the current game state
- **Move validation system** using python-chess to ensure legal moves
- **Multi-model architecture** combining the fine-tuned Gemma (opponent) with Gemini API (instructional chatbot)
- **Interactive web app** deployed on Hugging Face Spaces with Gradio

## Technical Implementation

### Model Fine-Tuning
- **Base Model**: Google Gemma 2B
- **Framework**: Keras-NLP with PyTorch backend
- **Training Data**: Chess games in algebraic notation (e.g., "e4 e5 Nf3 Nc6...")
- **Prompt Format**: Instruction-tuning template for next-move prediction
  ```
  Instruction: Predict the next chess move in the sequence ['e4', 'e5', 'Nf3']
  Response: Nc6
  ```

### Inference Strategy
- **Sampling**: TopK (k=50) with temperature 0.7 to balance creativity and valid move generation
- **Validation Loop**: Retry mechanism (up to 10 attempts) to handle invalid move predictions
- **State Management**: python-chess library for board state and move legality verification

### Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Gradio Interface                      │
├─────────────────┬───────────────────┬───────────────────┤
│  Chess Board    │   Chat Assistant  │ Openings Explorer │
│  (SVG Display)  │   (Gemini API)    │ (Lichess Dataset) │
├─────────────────┴───────────────────┴───────────────────┤
│              Fine-tuned Gemma 2B (Keras-NLP)            │
│                   Move Prediction Engine                 │
└─────────────────────────────────────────────────────────┘
```

## What I Learned

- **LLMs can learn chess patterns** but struggle with long-term strategy compared to traditional engines
- **Move validation is essential** — LLMs occasionally generate syntactically correct but illegal moves
- **Instruction-tuning format** helps the model understand the task structure
- **Temperature tuning** affects the balance between "creative" vs. predictable play
- **Multi-model architectures** can combine specialized models for different aspects of an application

## Skills Demonstrated

- **LLM Fine-Tuning**: Adapting pre-trained models for domain-specific tasks
- **Keras-NLP / PyTorch**: Model loading, compilation, and inference
- **Prompt Engineering**: Designing effective instruction templates
- **ML System Design**: Building end-to-end inference pipelines with validation
- **API Integration**: Google Gemini API, Hugging Face Datasets
- **Web App Development**: Gradio, deployment on Hugging Face Spaces

## Read More

I wrote detailed articles about the process:

- [Can Large Language Models Learn to Play Chess?](https://medium.com/@valentin.urena/can-large-language-models-learn-to-play-chess-e7151ecc140c) — Exploring the concept and initial findings
- [How to Fine-Tune a Large Language Model to Play Chess](https://medium.com/@valentin.urena/how-to-fine-tune-a-large-language-model-to-play-chess-6da5ee5ab986) — Technical deep-dive into the fine-tuning process

### Kaggle Notebooks
- [Play a Chess Match vs Gemma](https://www.kaggle.com/code/valentinbaltazar/play-a-chess-match-vs-gemma) — Inference and gameplay implementation
- [Train & Play Against Your Own Chess LLM](https://www.kaggle.com/code/valentinbaltazar/train-play-against-your-own-chess-llm) — Full training pipeline

## Resources

- **Fine-tuned Model**: [kaggle://valentinbaltazar/gemma-chess/keras/gemma_2b_en_chess](https://www.kaggle.com/models/valentinbaltazar/gemma-chess)
- **Chess Openings Dataset**: [Lichess/chess-openings](https://huggingface.co/datasets/Lichess/chess-openings)