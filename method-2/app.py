from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the pre-trained model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to calculate similarity between two text paragraphs
def calculate_similarity(text1, text2):
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    similarity = float(embeddings1.dot(embeddings2.T))
    return similarity

# Function to normalize similarity scores
def normalize_similarity(similarity):
    scaler = MinMaxScaler()
    normalized_similarity = scaler.fit_transform([[similarity]])[0][0]
    return normalized_similarity

@app.route('/predict_similarity', methods=['POST'])
def predict_similarity():
    try:
        data = request.get_json()
        text1 = data['text1']
        text2 = data['text2']

        similarity = calculate_similarity(text1, text2)
        normalized_similarity = normalize_similarity(similarity)

        response = {
            'similarity score': normalized_similarity
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
