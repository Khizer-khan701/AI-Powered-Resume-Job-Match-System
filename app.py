from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from src.helper import allowed_file, process_resume

UPLOAD_FOLDER = "uploads"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "resume" not in request.files:
            return "No file part"
        file = request.files["resume"]
        job_description = request.form["job_description"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            file.save(filepath)

            # Process resume
            result = process_resume(filepath, job_description)
            return render_template("result.html", result=result, job_description=job_description)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
