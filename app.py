from flask import Flask, request, render_template
from function import perform_text_summarization

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        summarized_text = perform_text_summarization(input_text)
        return render_template('index.html', input_text=input_text, summarized_text=summarized_text)
    return render_template('index.html', input_text= "", summarized_text="")

if __name__ == '__main__':
    app.run(debug=True)
