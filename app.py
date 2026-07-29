import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request
import joblib
import numpy as np

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)

app = Flask(__name__)

# Load trained model
model = joblib.load('housing_model.pkl')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Housing Price Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background-color: #343541; color: #ececf1; display: flex; flex-direction: column; height: 100vh; }
        .header { padding: 16px; background-color: #202123; border-bottom: 1px solid #4d4d4f; text-align: center; font-size: 1.2rem; font-weight: bold; }
        .chat-box { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
        .message { max-width: 80%; padding: 12px 16px; border-radius: 8px; line-height: 1.5; font-size: 0.95rem; }
        .user-message { background-color: #343541; align-self: flex-end; border: 1px solid #565869; }
        .bot-message { background-color: #444654; align-self: flex-start; }
        .input-area { padding: 20px; background-color: #343541; display: flex; gap: 10px; max-width: 800px; width: 100%; margin: 0 auto; }
        input { flex: 1; padding: 14px; border-radius: 6px; border: 1px solid #565869; background-color: #40414f; color: white; outline: none; font-size: 1rem; }
        button { padding: 14px 24px; border-radius: 6px; border: none; background-color: #19c37d; color: white; font-weight: bold; cursor: pointer; transition: background 0.2s; }
        button:hover { background-color: #1a7f5a; }
    </style>
</head>
<body>
    <div class="header">🏠 Housing Price Predictor AI</div>
    <div class="chat-box" id="chatBox">
        <div class="message bot-message">
            Hello! Enter all required features separated by commas.<br><br>
            <b>Order:</b> area, bedrooms, bathrooms, stories, mainroad (1/0), guestroom (1/0), basement (1/0), hotwaterheating (1/0), airconditioning (1/0), parking, prefarea (1/0),semi-furnished(0/1),unfurnished(0/1)
        </div>
    </div>
    <div class="input-area">
        <input type="text" id="userInput" placeholder="e.g. 7420, 4, 2, 3, 1, 0, 0, 0, 1, 2, 1" onkeydown="if(event.key==='Enter') sendMessage()" />
        <button type="button" onclick="sendMessage()">Send</button>
    </div>

    <script>
        async function sendMessage() {
            const inputField = document.getElementById('userInput');
            const chatBox = document.getElementById('chatBox');
            const userText = inputField.value.trim();

            if (!userText) return;

            // Display user message
            const userDiv = document.createElement('div');
            userDiv.className = 'message user-message';
            userDiv.textContent = userText;
            chatBox.appendChild(userDiv);
            inputField.value = '';

            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                // Map inputs to numbers (handling yes/no inputs smoothly)
                const featuresArray = userText.split(',').map(val => {
                    const cleanVal = val.trim().toLowerCase();
                    if (cleanVal === 'yes') return 1;
                    if (cleanVal === 'no') return 0;
                    return parseFloat(cleanVal);
                });

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: featuresArray })
                });

                const data = await response.json();
                const botDiv = document.createElement('div');
                botDiv.className = 'message bot-message';

                if (response.ok) {
                    let output = "<b>Predicted Price:</b> $" + Number(data.predicted_price).toLocaleString();
                    if (data.ai_explanation) {
                        output += "<br><br><b>AI Analysis:</b> " + data.ai_explanation;
                    }
                    botDiv.innerHTML = output;
                } else {
                    botDiv.textContent = "Error: " + (data.error || "Execution failed");
                }

                chatBox.appendChild(botDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            } catch (err) {
                const botDiv = document.createElement('div');
                botDiv.className = 'message bot-message';
                botDiv.textContent = 'Server communication error. Check your python terminal.';
                chatBox.appendChild(botDiv);
            }
        }
    </script>
</body>
</html>
"""

# NEW FUNCTION ADDED: Sends features and predicted price to Groq API
def generate_groq_explanation(raw_features, predicted_price):
    prompt = f"""
    You are a professional real estate expert. 
    Our Machine Learning model predicted a price of ${predicted_price:,.2f} for a house.

    Property Details (raw feature array):
    {raw_features}

    Task: Write a concise, professional 2-3 sentence real estate summary explaining why this property got this valuation based on key drivers like area, rooms, parking,amenities and last add (best regards M.I team).
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful and professional real estate consultant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Could not generate AI summary. Error: {str(e)}"


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        raw_features = data.get('features', [])

        if any(np.isnan(x) for x in raw_features):
            return (
                jsonify(
                    {'error': 'Invalid inputs. Ensure all values are numeric or yes/no.'}
                ),
                400,
            )

        features = np.array([raw_features])
        prediction = model.predict(features)
        predicted_price = round(float(prediction[0]), 2)

        # Call the new Groq function
        explanation = generate_groq_explanation(raw_features, predicted_price)

        return jsonify({
            'predicted_price': predicted_price,
            'ai_explanation': explanation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)