import os
import json

import google.generativeai as genai


working_directory = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_directory}/config.json"))
GOOGLE_API_KEY = config_data['GOOGLE_API_KEY']

genai.configure(api_key = GOOGLE_API_KEY)

def load_gemini_pro_model():
    gemini_pro_model = genai.GenerativeModel('gemini-pro')
    return gemini_pro_model

def load_gemini_pro_vision_model(prompt,image):
    gemini_pro_vision_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_pro_vision_model.generate_content([prompt,image])
    return response.text

def embeddings_response(text_input):
    embedding_model = 'models/text-embedding-004'
    embeddings = genai.embed_content(model = embedding_model,
                        content = text_input,
                        task_type="retrieval_document")
    embeddings_list = embeddings['embedding']
    return embeddings_list

def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel('gemini-pro')
    response = gemini_pro_model.generate_content(user_prompt)
    return response.text
