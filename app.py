from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pickle
import os
from dotenv import load_dotenv
from google import genai
from algo import mixedPrediction
import warnings
import pymongo
from pymongo.errors import BulkWriteError
import hashlib
load_dotenv()
# PROVIDE YOUR OWN MONGO DB CONNECTION STRING HERE ---------------------------
myclient = pymongo.MongoClient(f"mongodb+srv://{os.getenv('USER_DB')}:{os.getenv('PASSWORD_DB')}@{os.getenv('CLUSTER_URL')}/")
db = myclient["database"]
users = db["users"]

def hashText(key):
    hashObject = hashlib.sha256()
    hashObject.update(key.encode('utf-8'))
    hashedText = hashObject.hexdigest()
    return hashedText

# Note: THE PASSWORD PARAMETER IS A PASSWORD HASH   (UNCOMMENT THIS ON THE FIRST RUN TO INITIALISE THE DB)----------------
# user_list = [
#     { "username": "admin", "password": hashText("password")},
#     { "username": "Malachi", "password": hashText("password") }
# ]

# users.insert_many(user_list)

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="algo")


client = genai.Client(api_key=os.getenv("GEMINI_KEY_ONE"))

app = Flask(__name__)
app.secret_key = os.getenv('SUPER_SECRET_KEY')

@app.route('/')
def index():
    return redirect(url_for('form'))  # Redirect to the login page

@app.route('/home')
def home():
    #user = request.args.get('name', 'Guest')  # Get the name from query parameters, default to 'Guest'
    if 'user' in session:
        user = session['user']
        return render_template('index.html',name=user)
    else:
        return redirect(url_for('form'))  # Redirect to the login page if no user is logged in



# @app.route('/api/data', methods=['GET'])
# def get_data():
#     data = {
#         "message": "This is a sample response from the API.",
#         "status": "success"
#     }
#     return jsonify(data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        ticker = data.get('ticker')
        exog_features = data.get('exog_features', []) # Get exog_features, default to empty list
        
        if not ticker:
            return jsonify({'error': 'Ticker symbol is required'}), 400

        predicted_price, current_price, graph_base64 = mixedPrediction(ticker, exog_features=exog_features)

        if predicted_price is None and current_price is None and graph_base64 is None:
            return jsonify({'error': 'Prediction failed for the given ticker'}), 500

        return jsonify({
            'predicted_price': predicted_price,
            'current_price': current_price,
            'graph_base64': graph_base64
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['GET', 'POST']) # GET to display the form, POST to handle form submission
def form():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users.find_one({"username": username})
        if user and user["password"] == hashText(password):
            session['user'] = username
            return redirect(url_for('home'))
        else:
            error = "Invalid credentials, please try again."
            return render_template('form.html', error=error)

    return render_template('form.html')

"""def form():
    # if the request is a POST, process the form data
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # simple validation (add auth0 and other dbs later)
        if username=='admin' and password:
            session['user'] = username  # store the username in the session
            #return redirect(url_for('home', name=username)) # name=username will change the url to /home?name=username
            return redirect(url_for('home'))
        else:
            error = "Invalid credentials, please try again."
            return render_template('form.html', error=error)
    
    return render_template('form.html')"""

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('form'))  # Redirect to the login page after logout

@app.route('/greet/<name>')
def greet(name):
    return f"Hello, {name}!"

@app.route('/test')
def test():
    return render_template('test.html')

# Mock Gemini API for testing
"""@app.route('/gemini', methods=['POST'])
def gemini_mock():
    data = request.get_json()
    user_input = data.get('user_input', '').strip().lower()
    print(f"[MOCK] Received input: {user_input}")

    fake_map = {
        "apple": "AAPL",
        "aaaaple": "AAPL",
        "google": "GOOGL",
        "microsoft": "MSFT",
        "tesla": "TSLA"
    }

    if user_input in fake_map:
        return jsonify({'result': fake_map[user_input]})
    elif "brand" in user_input or "company" in user_input:
        return jsonify({'result': "No recognizable company mentioned. Please specify a publicly traded company."})
    else:
        return jsonify({'result': "Company specified is not publicly traded."})"""

# Gemini API for extracting stock ticker from user input
@app.route('/gemini', methods=['POST'])
def gemini():
        data = request.get_json()
        user_input = data['user_input']
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=(
                "Given the user input below, identify whether it refers to a known company.\n"
                "If the input contains a misspelling of a company name (e.g., 'aaaaple' instead of 'apple'), attempt to correct it.\n"
                "If it refers to a publicly traded company, return only its official stock ticker symbol in uppercase letters (e.g., 'AAPL' for Apple).\n"
                "If it refers to a known company that is not publicly traded — including private companies, local businesses, or organizations without stock listings — return: 'Company specified is not publicly traded.'\n"
                "If the input contains vague terms (e.g., 'tech companies', 'car brands') or no identifiable company name, return: 'No recognizable company mentioned. Please specify a publicly traded company.'\n"
                "Do not provide any explanations — return only the appropriate output message or ticker symbol.\n"
                "If the user enters a ticker symbol, return the same ticker symbol.\n"
                "If the user asks for a random company, use a random valid ticker symbol. dont respond with No recognizable company mentioned.\n"
                "If the user enters multiple ticker symbols/companies, return the ticker symbol of the first company only.\n"
                f"The user input: {user_input}"
            )
        )
        return jsonify({'result': response.text})

if __name__ == '__main__':
    app.run(port=6969,debug=True)
    #print(os.environ.get('TEST', 'No test value found'))
