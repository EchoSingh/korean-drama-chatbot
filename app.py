import pandas as pd
import gradio as gr
import torch
from sentence_transformers import SentenceTransformer, util

file_path = 'data/korean_drama.csv'
def load_data(file_path):
    """Load and preprocess the Korean drama data."""
    df = pd.read_csv(file_path)
    df['synopsis'] = df['synopsis'].astype(str)  
    return df

def create_embeddings(model, texts):
    """Create embeddings for the given texts using the provided model."""
    return model.encode(texts, convert_to_tensor=True)

def chatbot(query, model, drama_embeddings, korean_drama_df):
    """Chatbot function to find the most relevant dramas based on the query."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, drama_embeddings)[0]
    
    # Find the top 3 most similar dramas
    top_results = torch.topk(cosine_scores, k=3)
    
    response = ""
    for score, idx in zip(top_results[0], top_results[1]):
        row = korean_drama_df.iloc[idx.item()]
        response += f"**{row['drama_name']}**\n"
        response += f"Year: {row['year']}\n"
        response += f"Director: {row['director']}\n"
        response += f"Screenwriter: {row['screenwriter']}\n"
        response += f"Country: {row['country']}\n"
        response += f"Type: {row['type']}\n"
        response += f"Total Episodes: {row['tot_eps']}\n"
        response += f"Duration: {row['duration']} minutes\n"
        response += f"Start Date: {row['start_dt']}\n"
        response += f"End Date: {row['end_dt']}\n"
        response += f"Aired On: {row['aired_on']}\n"
        response += f"Original Network: {row['org_net']}\n"
        response += f"Content Rating: {row['content_rt']}\n"
        response += f"Synopsis: {row['synopsis']}\n"
        response += f"Rank: {row['rank']}\n"
        response += f"Popularity: {row['pop']}\n"
        response += "\n\n"
    
    return response

def main():
    # Load data
    data_file = 'data/korean_drama.csv'
    korean_drama_df = load_data(data_file)

    # Load the pre-trained SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings for the drama synopses
    drama_synopses = korean_drama_df['synopsis'].tolist()
    drama_embeddings = create_embeddings(model, drama_synopses)

    # Define the Gradio interface
    iface = gr.Interface(
        fn=lambda query: chatbot(query, model, drama_embeddings, korean_drama_df),
        inputs="text",
        outputs="text",
        title="Korean Drama Chatbot",
        description="Ask me about Korean dramas!"
    )

    # Launch the interface
    iface.launch()

if __name__ == "__main__":
    main()
