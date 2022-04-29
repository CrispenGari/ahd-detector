"""
* note that you only need to download the tokenizer model once from spacy.
* also you need to download "" once from nltk
"""
# import spacy, nltk
# spacy.cli.download("en_core_web_sm")
# nltk.download('punkt')

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from app import app
from flask import make_response, jsonify
from blueprints import blueprint
from flask_graphql import GraphQLView
from schema import schema

app.register_blueprint(blueprint, url_prefix="/api")
class AppConfig:
    PORT = 3001
    DEBUG = False
    
@app.route('/', methods=["GET"])
def meta():
    meta ={
        "programmer": "@crispengari",
        "main": "automatic humour detection(ahd)",
        "description": "given a text detect if there's humour or not in that given text.",
        "language": "python",
        "libraries": ["pytorch", "tensorflow", "keras", "torchtext"],
    }
    return make_response(jsonify(meta)), 200

app.add_url_rule('/graphql', view_func=GraphQLView.as_view(
    'graphql',
    schema=schema,
    graphiql=True,
))

if __name__ == "__main__":
    app.run(debug=AppConfig().DEBUG, port=AppConfig().PORT, )