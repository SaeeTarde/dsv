import numpy as np
from sklearn.neural_network import MLPClassifier

X=np.array([[0,0],[0,1],[1,0],[1,1]])

and_gate=np.array([0,0,0,1])
nand_gate=np.array([1,1,1,0])
or_gate=np.array([0,1,1,1])

and_model=MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='tanh',
    max_iter=10000
)

and_model.fit(X,and_gate)

nand_model=MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='tanh',
    max_iter=10000
)

nand_model.fit(X,nand_gate)

or_model=MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='tanh',
    max_iter=10000
)

or_model.fit(X,or_gate)

and_pred=and_model.predict(X)
nand_pred=nand_model.predict(X)
or_pred=or_model.predict(X)

print("AND Prediction:",and_pred)
print("NAND Prediction:",nand_pred)
print("OR Prediction:",or_pred)