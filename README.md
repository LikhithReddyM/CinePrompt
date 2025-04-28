## CinePrompt - Personalized Movie Recommender with RAG and LLM's

This is a movie recommendation system that uses RAG (Retrieval-Augmented Generation) and LLM to provide personalized movie recommendations based on user preferences.

## Setup Instructions
After cloning this repository, please download the `saved_data` folder from the following Google Drive link: https://drive.google.com/drive/folders/1ij0tpnAsUv6GqkWCGxhn3DxntN8vzInr

Note: Due to GitHub's storage limitations, the saved_data directory is provided separately via Google Drive

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the movies dataset from Kaggle and place it in the project directory as `movies.csv`

3. Download a Llama model in GGUF format and update the model path in `app.py`:
```python
llm = Llama(
    model_path="path/to/your/llama-model.gguf",  # Update this path
    n_ctx=2048,
    n_threads=4
)
```

4. Run the Flask application:
```bash
python app.py
```

5. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Enter your movie preferences in the search box (e.g., "humorous and thriller movies")
2. Click "Get Recommendations" to see the top 20 recommended movies
3. The results will show movie titles and their IDs

## How It Works

1. The system uses sentence-transformers to create embeddings for each movie in the dataset
2. When a user query is received, it's converted to an embedding
3. FAISS is used to find the most similar movies based on the query embedding
4. The relevant movie information is used to create a prompt for the LLM
5. The LLM generates the final recommendations based on the retrieved information

## Requirements

- Python 3.8+
- Flask
- Pandas
- sentence-transformers
- faiss-cpu
- llama-cpp-python
- numpy

## Flowchart

<img width="979" alt="flowchart" src="https://github.com/user-attachments/assets/9babbd79-e60f-403c-915f-2c0bb7f9d515" />
