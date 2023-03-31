import numpy as np
import matplotlib.pyplot as plt


# Carga de datos
data = np.genfromtxt('framingham.csv', delimiter=',', skip_header=1)

# Eliminando filas con valores NaN
data = data[~np.isnan(data).any(axis=1)]


# Variable independiente - Nivel de colesterol total
X = data[:, 9]
# Variable dependiente - Riesgo de enfermedad coronaria en 10 anios
y = data[:, 15]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = -(1/m) * (np.sum(y * np.log(h) + (1-y) * np.log(1-h)))
    return J

def gradient(X, y, theta):
    m = len(y)
    n = X.shape[1]  
    h = sigmoid(np.dot(X, theta))
    return np.array([np.sum((h - y) * X[:, j]) for j in range(n)])/m


# Normalizacion de datos
X = (X - np.mean(X)) / np.std(X)

# Agregar columna de unos
X = np.column_stack((np.ones(X.shape[0]), X))

# Inicializacion de parametros
theta = np.zeros(X.shape[1])


def logistic_reg(X, y, t, a, n):
    costs = []
    for i in range(n):
        t -= a * gradient(X, y, t)
        costs.append(cost(X, y, t))
    return t, costs  

# Entrenamiento del modelo
new_theta, costs = logistic_reg(X, y, theta, a=0.0001, n=500)

print('Theta: ', new_theta)


# Grafica de la funcion de costo
plt.plot(costs)
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.title('Regresion logistica')
plt.show()

# Grafica de la funcion de regresion logistica
xx = np.arange(X[:, 1].min(), X[:, 1].max() + 0.1, 0.1)
yy = sigmoid(xx)
plt.scatter(X[:, 1], y, color='red')
plt.plot(xx, yy) 
plt.xlabel('Nivel de colesterol total')
plt.ylabel('Riesgo de enfermedad coronaria en 10 años')
plt.title('Regresion logistica')
plt.show()


# Cross validation
def cross_val(X, y, k):
    
    p = np.arange(len(y))
    np.random.seed(123)
    np.random.shuffle(p)
    X = X[p]
    y = y[p]
    
    train_size = int(0.7 * len(y))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    k_times = len(y_train) // k
    test_errors, training_errors = [], []
    
    for i in range(k):
        # Dividir datos en train y test
        x_train_k = np.concatenate((X_train[:i*k_times], X_train[(i+1)*k_times:]))
        y_train_k = np.concatenate((y_train[:i*k_times], y_train[(i+1)*k_times:]))
        x_test_k = X_train[i*k_times:(i+1)*k_times]
        y_test_k = y_train[i*k_times:(i+1)*k_times]
        
        theta_0 = np.zeros(x_train_k.shape[1])
        theta_f = logistic_reg(x_train_k, y_train_k, theta_0, a=0.0001, n=1000)[0]
        
        # Calcular error de entrenamiento
        y_pred = sigmoid(np.dot(x_train_k, theta_f))
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        training_errors.append(np.mean(y_pred != y_train_k))

        # Calcular error de test
        y_pred = sigmoid(np.dot(x_test_k, theta_f))
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        test_errors.append(np.mean(y_pred != y_test_k))

    mean_errors = [(training_errors[i] + test_errors[i]) / 2 for i in range(k)]
    best_degree = mean_errors.index(min(mean_errors))
    mean_error = min(mean_errors)
    
    return best_degree, 1-mean_error

k = 20
print('Mejor grado polinomial: ', cross_val(X, y, k)[0], " con un accuracy de ", cross_val(X, y, k)[1]*100, "%")
