import os

from flask import Flask, render_template, request, session, redirect

from utils import load_documents, create_chunks, create_embeddings
from utils import similarity_search, question_answering

app = Flask(__name__)
app.secret_key = 'Ask questions from your documents'

# to store the last five questions asked by the user
history = []

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get the file from the request
        file = request.files["file"]
        service_types = request.form.getlist("services")[0]
        
        # store the service type in the session
        session["service_type"] = service_types
        # if the file is not empty, then save it to the uploads folder
        if file:
            # Create the uploads folder if it does not exist
            folder_path = os.path.join("uploads")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # Save the file to the uploads folder
            file_path = os.path.join(folder_path, file.filename)
            session["file_path"] = file_path
            file.save(os.path.join(folder_path, file.filename))

            # redirect to the question-answering page
            return redirect("/ask-question")
        else:
            # If the file does not exist, then render the template with the file upload message
            return render_template("index.html", render_alert = "Please upload a file first.")
    elif request.method == "GET":
        # render the default home page
        return render_template("index.html", render_alert = False)
    
@app.route("/ask-question", methods=["GET", "POST"])
def ask_question():
    # get the file path from the session
    file_path = session.get("file_path", 'Default Value')
    service_type = session.get("service_type", "Default Value")

    # if the file path is not set, then redirect to the home page
    if file_path == 'Default Value':
        return redirect("/")
    elif service_type == "Default Value":
        return redirect("/")
    
    # load the document in the Langchain
    documents = load_documents(file_path)
    # create chunks from the documents
    chunks = create_chunks(documents)
    # create embeddings
    db = create_embeddings(chunks, service_type=service_type)

    if request.method == "POST":
        # get the most similar chunk to the user's query
        query = request.form["query"]
        docs = similarity_search(db, query)
        # find the answer of the user input's query
        response = question_answering(docs, query, service_type=service_type)

        # store the question and the answer in the history
        history.append({"question": query, "answer": response})
        # if the history is more than 5, then remove the first element
        if len(history) > 5:
            history.pop(0)
        return render_template("question-answering.html", response=response, history=history)
    else:
        # render the default home page
        return render_template("question-answering.html")


if __name__ == "__main__":
    print(f"PORT={os.environ.get('PORT', 8080)}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
