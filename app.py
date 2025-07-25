from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pickle
import os
from dotenv import load_dotenv
load_dotenv()

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



@app.route('/api/data', methods=['GET'])
def get_data():
    data = {
        "message": "This is a sample response from the API.",
        "status": "success"
    }
    return jsonify(data)

@app.route('/login', methods=['GET', 'POST']) # GET to display the form, POST to handle form submission
def form():
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
            return "Invalid credentials, please try again.", 401
    
    return render_template('form.html')

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

if __name__ == '__main__':
    app.run(port=6969,debug=True)
    #print(os.environ.get('TEST', 'No test value found'))
