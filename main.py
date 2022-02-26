import numpy as np
#import matplotlib.pyplot
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    # learning rate is 0.3
    learning_rate = 0.3

    # create instance of neural network
    neuronka = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    nahodnecislo = np.random.rand(3, 3) - 0.5
    print(nahodnecislo)

    # download MNIST 100
    data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
    data_list = data_file.readlines()
    data_file.close()
    print(len(data_list))
    print(data_list[0])

    #for i in range(90):
    # idem si vykrelit cislo nacitane z csv
    all_values = data_list[0].split(',')  # split long text separated by commas
    image_array = np.asfarray(all_values[1:]).reshape(
        (28, 28))  # ignore the first value which is label and take remaining -> 28x28
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()

    #Preparing the MNIST Training data 151
    # vsetky hodnoty pixelov chcem dostat z intervalu 0-255 do 0.01 az 1.00
    print(image_array)
    scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    print(scaled_input)