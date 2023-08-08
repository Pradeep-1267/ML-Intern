from teeth_shade import shade_predictor


predictor = shade_predictor()

result = predictor.predict("./yy.jfif" )

#result contains list of two objects, first is an imteger 0 , 1 or 2 denoting the class of the image
#class 0: White teeth
#class 1: Whitish yellow teeth
#class 2: yellow teeth
#second object is an array having probabilities of each class

print(result)