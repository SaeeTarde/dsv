import numpy as np
from sklearn.neural_network import MLPClassifier

X=np.array([[0,0],[0,1],[1,0],[1,1]])

xor=np.array([0,1,1,0])
xnor=np.array([1,0,0,1])

nor_model=MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='tanh',
    max_iter=10000
)

nor_model.fit(X,xnor)

xor_model=MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='tanh',
    max_iter=10000
)

xor_model.fit(X,xor)

nor_pred = nor_model.predict(X)
xor_pred = xor_model.predict(X)

print("XOR Prediction:")
print(xor_pred)

print("XNOR Prediction:")
print(nor_pred)