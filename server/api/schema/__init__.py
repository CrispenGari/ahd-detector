import graphene

from models.pytorch import predict_humour, ahd_model

class PredictionInputType(graphene.InputObjectType):
    """
    This is the input object that that the user will pass when making queries to the graphql
    server:
        model_name:str = "tf-model" or "pt-model" (tensorflow or pytorch model)
        text:str = The sentence or text which the user want to predict humour
    """
    modelType = graphene.String(required=True, default_value="pt")
    text = graphene.String(required=True)
    
class PredictionType(graphene.ObjectType):
    """
    This a prediction response object from the model.
    """
    label = graphene.Int(required=True)
    probability = graphene.Float(required=True)
    class_ = graphene.String(required=True)
    text = graphene.String(required=True)

class ErrorType(graphene.ObjectType):
    """
    This is the error type
    """
    field = graphene.String(required=True)
    message = graphene.String(required = True)

class PredictionResponse(graphene.ObjectType):
    """
    This class object is the object type that will return the
    prediction response data that we are interested in
    """
    error = graphene.Field(ErrorType, required=False)
    ok = graphene.Boolean(required=True)
    prediction = graphene.Field(PredictionType, required=False)
    
    
class Query(graphene.ObjectType):
    predict_humour = graphene.Field(graphene.NonNull(PredictionResponse),
                                    input=graphene.Argument(graphene.NonNull(PredictionInputType))
                                    )
    def resolve_predict_humour(root, info, input):
        model = "tensorflow" if str(input.get('modelType')).lower().strip() == "tf" else "pytorch"
        if model == "tensorflow":
            pass
        else:
            res = predict_humour(input.get('text'), model=ahd_model)
        return PredictionResponse(
            ok=True,
            prediction=PredictionType(
                label=res.label,
                probability=res.probability,
                class_=res.class_,
                text= res.text
            )
        )
    
schema = graphene.Schema(query=Query)