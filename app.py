from flask import Flask, jsonify, render_template_string, request
import joblib
import numpy as np

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
            <b>Order:</b> area, bedrooms, bathrooms, stories, mainroad (1/0), guestroom (1/0), basement (1/0), hotwaterheating (1/0), airconditioning (1/0), parking, prefarea (1/0)
        </div>
    </div>
    <div class="input-area">
        <input type="text" id="userInput" placeholder="e.g. 7420, 4, 2, 3, 1, 0, 0, 0, 1, 2, 1" />
        <button onclick="sendMessage()">Send</button>
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
                // Map values: Converts "yes"/"no" to 1/0 or parses numbers
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
                    botDiv.textContent = Predicted Price: $${data.predicted_price.toLocaleString()};
                } else {
                    botDiv.textContent = Error: ${data.error};
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


@app.route('/')
def home():
  return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
  try:
    data = request.get_json()
    raw_features = data.get('features', [])

    # Check for NaN values
    if any(np.isnan(x) for x in raw_features):
      return (
          jsonify(
              {'error': 'Invalid inputs. Ensure all values are numeric or yes/no.'}
          ),
          400,
      )

    # Convert to 2D NumPy array
    features = np.array([raw_features])

    # Run Prediction
    prediction = model.predict(features)
    return jsonify({'predicted_price': round(float(prediction[0]), 2)})

  except Exception as e:
    # Print real error trace to terminal
    print('Prediction Exception:', str(e))
    return jsonify({'error': str(e)}), 400


if __name__ == '_main_':
  app.run(debug=True)