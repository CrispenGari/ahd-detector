

class WrongHttpMethodException(Exception):
    pass

class RequiredQueryStringException(Exception):
     pass

class InvalidQueryStringException(Exception):
    pass

class EmptyJsonBodyException(Exception):
    pass
