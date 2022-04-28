"""
note that you only need to download the tokenizer model once from spacy.
"""
# import spacy
# spacy.cli.download("en_core_web_sm")

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
    DEBUG = True
    
    
@app.route('/', methods=["GET"])
def meta():
    meta ={
        "programmer": "@crispengari",
        "main": "audio classification",
        "description": "given an a audio of an animal the model should classify weather the sound is for a cat or a dog.",
        "language": "python",
        "library": "pytorch",
        "mainLibray": "torchaudio"
    }
    return make_response(jsonify(meta)), 200

app.add_url_rule('/graphql', view_func=GraphQLView.as_view(
    'graphql',
    schema=schema,
    graphiql=True,
))

if __name__ == "__main__":
    app.run(debug=AppConfig().DEBUG, port=AppConfig().PORT, )