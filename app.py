from flask import Flask, request, jsonify, send_from_directory
import os
import google.generativeai as genai
import json

# Configure the Gemini API
# IMPORTANT: Set the GEMINI_API_KEY environment variable to your actual Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not set. Gemini API functionality will be disabled.")
    GEMINI_API_KEY = "DUMMY_KEY" # Set a dummy key to avoid errors
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/api/analyze', methods=['POST'])
def analyze_file():
    """Analyzes a file using the Gemini API."""
    data = request.get_json()
    filepath = data.get('filepath')

    if not filepath:
        return jsonify({'error': 'Filepath is required.'}), 400

    # Sanitize the filepath to prevent directory traversal
    base_dir = os.path.abspath(os.path.dirname(__file__))
    safe_path = os.path.abspath(os.path.join(base_dir, filepath))

    if not safe_path.startswith(base_dir) or not os.path.exists(safe_path):
        return jsonify({'error': 'Invalid or non-existent file path.'}), 404

    with open(safe_path, 'r') as f:
        content = f.read()

    # Call the Gemini API
    if GEMINI_API_KEY == "DUMMY_KEY":
        analysis = {
            'explanation': f'This is a placeholder for the analysis of {os.path.basename(filepath)}. Real Gemini API integration is required.',
            'suggestions': [],
            'new_inventions': []
        }
    else:
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = (
                "Analyze the following Python code and provide an explanation, a list of suggestions for improvement, "
                "and a list of new inventions based on the code's concepts. Your response MUST be a valid JSON object "
                "with the following keys: 'explanation' (a string), 'suggestions' (a list of strings), and "
                f"'new_inventions' (a list of strings).\n\nCode:\n```python\n{content}\n```"
            )
            response = model.generate_content(prompt)

            # Clean the response to ensure it's valid JSON
            cleaned_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            analysis = json.loads(cleaned_text)

        except Exception as e:
            return jsonify({'error': f'An error occurred with the Gemini API: {e}'}), 500

    return jsonify(analysis)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
