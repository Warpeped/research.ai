# api/app.py
import os
import requests
from flask import Flask, request, jsonify
from groq import Groq
from bs4 import BeautifulSoup

app = Flask(__name__)

# Initialize the Groq client
# The API key will be read from the Vercel environment variables
try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    groq_client = None

@app.route('/api/research', methods=['POST'])
def handle_research():
    if not groq_client:
        return jsonify({"error": "Groq client not initialized. Check API key."}), 500

    data = request.get_json()
    url_to_scrape = data.get('url')
    user_query = data.get('query')

    if not url_to_scrape or not user_query:
        return jsonify({"error": "Missing 'url' or 'query' field"}), 400

    try:
        # Step 1: Scrape the content from the provided URL
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        page_response = requests.get(url_to_scrape, headers=headers, timeout=10)
        page_response.raise_for_status()
        
        soup = BeautifulSoup(page_response.content, 'html.parser')
        # Extract text from all paragraphs to get the main content
        page_text = ' '.join(p.get_text() for p in soup.find_all('p'))

        if not page_text:
            return jsonify({"summary": "Could not extract readable text from the provided URL."})

        # Step 2: Use the AI to process the scraped text
        system_prompt = "You are an expert research assistant. Based on the provided text, answer the user's query concisely and accurately."
        # Limit text to 8000 characters to avoid exceeding model context limits
        user_prompt = f"User Query: \"{user_query}\"\n\nProvided Text:\n---\n{page_text[:8000]}\n---" 

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama3-8b-8192",
        )
        summary = chat_completion.choices[0].message.content

        return jsonify({"summary": summary})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch or scrape the URL: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500
        
if __name__ == '__main__':
    app.run(debug=True)