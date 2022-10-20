from flask import Flask
import sys
app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello Flask!"

@app.route('/map')
def map():
    return '{msg:"Welcome to map"}'

@app.route('/1')
def one():
    return '{msg:"1 Page"}'

@app.route('/2')
def two():
    return '{msg:"2 Page"}'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(sys.argv[1]))
