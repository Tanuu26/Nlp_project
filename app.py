from flask import Flask, render_template, request
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from gingerit.gingerit import GingerIt

app = Flask(__name__)
model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
ginger_parser = GingerIt()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():
    if request.method == "POST":
        inputtext = request.form["inputtext_"]
        input_text = "summarize: " + inputtext
        tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512).to(device)
        summary_ = model.generate(tokenized_text, min_length=30, max_length=300)
        summary = tokenizer.decode(summary_[0], skip_special_tokens=True)

        # Grammar checking
        grammar_checked_summary = ginger_parser.parse(summary)["result"]

    return render_template("output.html", data={"summary": grammar_checked_summary})

if __name__ == '__main__':
    app.run()
