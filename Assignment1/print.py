import pickle

favorite_color = pickle.load( open( "params.pickle", "rb" ) )
print(favorite_color)
