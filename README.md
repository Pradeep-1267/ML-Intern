# ML-Intern
Developing ML Detectors
This model, given an input image classifies into one of the three classes, 0, 1 or 2
class 0: White teeth
class 1: Whitish yellow teeth
class 2: yellow teeth
There is a class shade_predictor in teeth_shade.py python script
this needs to be initialized, without passing any parameters 
example: predictor = shade_predictor()
after you create an instance, any image of teeth can be predicted using predict method of this class
example: result = predictor.predict(path)
result is a list of two objects, first object is the class ( 0, 1 or 2) and second object is a numpy array of 3 float32 numbers which are probabilities of each class
Architecture of the model is specified in the training notebook
