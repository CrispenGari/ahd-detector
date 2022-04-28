from flask import Blueprint, make_response, jsonify, request
from models.pytorch import predict_homour, ahd_model
from exceptions import *

blueprint = Blueprint("blueprint", __name__)

@blueprint.route('/detect-humour', methods=["POST"]) 
def detect_humour():
    MODEL_TYPE = None
    try:
        if request.query_string:
            qs = dict(request.args)
            if 'model' in qs.keys():
                if qs['model'].lower() == 'tf' or qs['model'].lower() == 'pt':
                    MODEL_TYPE = "tensorflow" if qs.get('model').lower() == 'tf' else 'pytorch'
                else:
                    raise InvalidQueryStringException(
                        "Values for query string 'model' can be either 'tf' or 'pt'"
                    ) 
            else:
               raise RequiredQueryStringException("the query string 'model' is required when making this request.") 
        else:
            raise RequiredQueryStringException("the query string 'model' is required when making this request.")
        if request.method == "POST":
            res = request.get_json(force=True)
            if res.get("text"):
                print(MODEL_TYPE)
                pred = predict_homour(res.get("text"), ahd_model)
                print(pred)
                return make_response(jsonify(pred.to_json())), 200
            else:
                raise EmptyJsonBodyException("you should pass the 'text' in your json body while making this request.")
        else:
            raise WrongHttpMethodException("the request method should be post only.")
    except Exception as e:
        print(e)
        return make_response(jsonify({
           "message": "internal server error.",
            "code": 500,
            "error": str(e)
        })), 500
        
    