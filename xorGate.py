import tensorflow as tf

# los programas tensorflow estan divididos en 3 partes
# por convencion mia en la primer parte definimos los datos de entrada que son los placeholders
# en la segunda parte definimos las variables que va a usar el programa internamente
# en la tercera parte definimos el modelo
# en la parte extra definimos el entrenamiento del modelo
# y en la parte extra 2 definimos una comprobacion de los datos

#las compuertas xor tienen la siguiente forma
# X | Y | Out
# 0 | 0 |  0
# 0 | 1 |  1
# 1 | 0 |  1
# 1 | 1 |  0

# para implementar un perceptron se hace y = w * x + b
# como w -> pesos [weights]  x -> matriz de entrada  b -> sesgo

# Definicion de datos de entrada
# shape define la matriz de entrada [filas, columnas]
x_ = tf.placeholder(tf.float32, [None, 2], name="x-input")
# datos de la salida ideal a las entradas en orden coherente
y_ = tf.placeholder(tf.float32,  [None, 1], name="y-input")


# w son los pesos en este caso inicializamos en cero y serian [cantidad_elementos, numero_de_perceptrones]
w0 = tf.Variable(tf.random_normal([2, 2], -1, 1), name="weights0")

# w en la segunda capa tiene 1 solo perceptron
w1 = tf.Variable(tf.random_normal([2, 1], -1, 1), name="weights1")

# b -> cantidad de salidas -> cantidad de perceptrones en la capa
b0 = tf.Variable(tf.random_normal([2]), name="biases0")
b1 = tf.Variable(tf.random_normal([1]), name="biases1")

#definicion del modelo en este caso es una sola capa
h1 = tf.nn.sigmoid(tf.add(tf.matmul(x_, w0), b0))
output = tf.nn.sigmoid(tf.add(tf.matmul(h1, w1), b1))

# ahora entrenamos la red reuronal
# definimos la funcion de error (min_square_error)
mse = tf.reduce_mean(tf.square(output - y_))

# le decimos a tensorflow que nos minimice el error
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(mse)

##################### aca ya terminamos de definir el modelo ###########################################


# ahora inicializamos la session de tensorflow
init = tf.global_variables_initializer()

# creamos una sesion
sess = tf.Session()
sess.run(init)

# definimos los datos de entrada
batchX = [[0,0], [0,1], [1,0], [1,1]]
batchY = [[0], [1], [1], [0]]

# ahora entrenamos el modelo
for i in range(200000):
    sess.run(train_step, feed_dict={x_: batchX, y_: batchY})
    loss1 = sess.run(mse, feed_dict={x_: batchX, y_: batchY})
    if i % 1000 == 0:
        print('Epoch ', i)
        print('out ', sess.run(output, feed_dict={x_: batchX, y_: batchY}))
        print('W0 ', sess.run(w0))
        print('Bias0 ', sess.run(b0))
        print('W1 ', sess.run(w1))
        print('Bias1 ', sess.run(b1))
        print('loss', loss1)

# fin del entrenamiento
print("Training finish")