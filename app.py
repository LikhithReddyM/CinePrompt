from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

DATA_FILE = 'data.json'

def load_data(): #load data from json file
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
        with open(DATA_FILE, 'w') as f:
            json.dump([], f)
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def save_data(entry): #save data to json file
    data = load_data()
    username = entry.get("username")

    if not username: # No user name
        return
    existing_user = next((user for user in data if user.get("username") == username), None)

    if existing_user: # existing user
        for key, value in entry.items():
            if key != "username":
                existing_user[key] = value
    else:
        # First-time user
        data.append(entry)

    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

@app.route('/')
def home(): # home page
    return render_template('index.html')

@app.route('/preferences/<username>') 
def preferences(username): # preferences page
    return render_template('preferences.html', username=username)

@app.route('/submit-preferences', methods=['POST'])
def submit_preferences(): # submit preferences
    try:
        user_data = request.get_json()
        if not user_data.get("username"): # No user name
            return jsonify({'status': 'fail', 'message': 'Username is required'}), 400

        save_data(user_data)
        return jsonify({'status': 'success', 'message': 'Preferences saved successfully'})
    except Exception as e:
        print("Error in submit-preferences route:", e)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/submissions')
def view_submissions(): # view submissions page
    data = load_data()
    return render_template('submissions.html', submissions=data)

if __name__ == '__main__':
    app.run(debug=True)
