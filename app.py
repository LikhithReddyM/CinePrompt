from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os, json, openai, logging

# ---------- Configuration ----------

load_dotenv()
OPENAI_API_KEY = "YOUR_API_KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")

app       = Flask(__name__)
DATA_FILE = "data.json"
logging.basicConfig(level=logging.INFO)

# ---------- Data helpers ----------
def load_data():
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE)==0:
        with open(DATA_FILE,"w") as f: json.dump([],f)
    with open(DATA_FILE) as f: return json.load(f)

def save_or_update(record):
    data = load_data()
    user = next((u for u in data if u["username"]==record["username"]), None)
    if user: user.update({k:v for k,v in record.items() if k!="username"})
    else:    data.append(record)
    with open(DATA_FILE,"w") as f: json.dump(data,f,indent=4)

# ---------- GPT helper ----------
def top5_movies(prompt: str) -> str:
    """Return a newline‑separated list of 5 movie recommendations."""
    q = (f"Suggest exactly five movie titles for this user prompt. "
         f"Return each title on its own line, nothing else.\n\nUser prompt: {prompt}")
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content": q}],
        temperature=0.7
    )
    return resp["choices"][0]["message"]["content"].strip()

# ---------- Routes ----------
@app.route("/")
def home(): return render_template("index.html")

@app.route("/preferences/<username>")
def preferences(username): return render_template("preferences.html",
                                                 username=username)

@app.route("/submit-preferences", methods=["POST"])
def submit_preferences():
    try:
        p = request.get_json(force=True)
        username = p.get("username","").strip()
        if not username:
            return jsonify(status="fail", message="Username required"), 400

        prompt = p.get("prompt","").strip() or "Give me five great movies."
        suggestions = top5_movies(prompt)

        record = {**p, "suggestions": suggestions}
        save_or_update(record)

        return jsonify(status="success",
                       message="Saved.",
                       suggestions=suggestions)

    except Exception as e:
        app.logger.error("submit-preferences error: %s", e)
        return jsonify(status="error", message=str(e)), 500

@app.route("/submissions")
def submissions(): return render_template("submissions.html",
                                          submissions=load_data())

if __name__ == "__main__":
    app.run(debug=True)
