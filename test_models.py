from models import perceptron, mlp, rbf

print("Perceptron:", callable(perceptron.train), callable(perceptron.predict))
print("MLP:", callable(mlp.train), callable(mlp.predict))
print("RBF:", callable(rbf.train), callable(rbf.predict))
