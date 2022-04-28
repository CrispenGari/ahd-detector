from flask import Blueprint, make_response, jsonify, request
from models.pytorch import predict_humour as torch_predict_humour, ahd_model as torch_model
from models.tensorflow import predict_humour as tf_predict_humour, ahd_model as tf_model
from exceptions import *

blueprint = Blueprint("blueprint", __name__)

@blueprint.route('/detect-humour', methods=["POST"]) 
def detect_humour():
    MODEL_TYPE = None
    try:
        if request.query_string:
            qs = dict(request.args)
            if 'model' in qs.keys():
                if str(qs['model']).lower() == 'tf' or str(qs['model']).lower() == 'pt':
                    MODEL_TYPE = "tensorflow" if str(qs.get('model')).lower() == 'tf' else 'pytorch'
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
                if MODEL_TYPE == "tensorflow":
                    pred = tf_predict_humour(res.get("text"), tf_model)
                else:
                    pred = torch_predict_humour(res.get("text"), torch_model)
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
        
    