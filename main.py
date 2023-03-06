from flask import Flask
from flask import render_template, request
from time import sleep

from summarizer import summarizer, readScore
from gpt3 import gpt3_summarizer, gpt3_readScore
from bart import bartSumm


app = Flask(__name__)

@app.route("/")
def hello_world():
    # return "<p>Hello, World!</p>"
    return render_template('index.html')

@app.route("/summarize", methods=["POST"])
def summarize():
    if request.method == 'POST':
        # print(request.headers)        
        # print(request.form['textInput'])
        # print(request.form['summaryDegree'])

        model = request.form['model']

        textInput = request.form['textInput']
        summaryDegree = float(request.form['summaryDegree'])

        # return redirect(url_for("success"))
        # sleep(5)
        if model == 'tf_idf':
            summary, originalLen, summaryLen = summarizer(textInput, summaryDegree)
            rScore = readScore(textInput)
        
        elif model == 'bart':
            summary, originalLen, summaryLen = bartSumm(textInput)
            rScore = readScore(textInput)

        elif model == 'gpt3':
            summary, originalLen, summaryLen = gpt3_summarizer(textInput, summaryDegree)
            rScore = gpt3_readScore(textInput)

        return {"summary": summary, "originalLen": originalLen, "summaryLen": summaryLen, "readScore": rScore}
    return render_template("index.html", )

if __name__ == '__main__':
    app.run()