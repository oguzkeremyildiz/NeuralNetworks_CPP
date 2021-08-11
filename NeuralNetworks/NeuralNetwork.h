//
// Created by Oğuz Kerem Yıldız on 10.08.2021.
//

#ifndef NEURALNETWORKS_CPP_NEURALNETWORK_H
#define NEURALNETWORKS_CPP_NEURALNETWORK_H
#include "ActivationFunctions/Activation.h"
#include "ActivationFunctions/ActivationFunction.h"
#include "ActivationFunctions/Sigmoid.h"
#include "ActivationFunctions/Linear.h"
#include "ActivationFunctions/ReLU.h"
#include "ActivationFunctions/TanH.h"
#include "Bias.h"
#include "Layer.h"
#include "InstanceList.h"
#include <utility>
#include <vector>

using namespace std;

class NeuralNetwork {
private:
    Layer* layers;
    int layersLength;
    InstanceList instanceList;
    int seed;
    ActivationFunction* function;
    Bias* biases;
    void createInputVector(Instance instance);
    void feedForward();
    vector<vector<vector<double>>> backpropagation(int classInfo, double learningRate, double momentum, vector<vector<vector<double>>> oldDeltaWeight);
    void calculateRMinusY(vector<vector<vector<double>>>& deltaWeights, int classInfo, double learningRate);
    vector<vector<double>> multiplyMatrices(vector<vector<double>> matrix1, vector<vector<double>> matrix2);
    double dotProduct(vector<double> vec, vector<vector<double>> matrix, int j);
    void setWeights(vector<vector<vector<double>>> deltaWeights, vector<vector<vector<double>>> oldDeltaWeights, double momentum);
    vector<vector<double>> calculateError(int i, vector<vector<vector<double>>>& deltaWeights);
    vector<vector<double>> hadamardProduct(vector<vector<double>> matrix1, vector<vector<double>> matrix2);
public:
    virtual ~NeuralNetwork();
    NeuralNetwork(int seed, vector<int>& hiddenLayers, InstanceList instanceList, Activation activation);
    void train(int epoch, double learningRate, double etaDecrease, double momentum);
    string predict(Instance instance);
    double test(InstanceList list);
};

NeuralNetwork::~NeuralNetwork() {
    delete[] layers;
    delete[] biases;
}

NeuralNetwork::NeuralNetwork(int seed, vector<int>& hiddenLayers, InstanceList instanceList, Activation activation) {
    ActivationFunction* function;
    switch (activation) {
        case Activation::SIGMOID:
            function = new Sigmoid();
            break;
        case Activation::RELU:
            function = new ReLU();
            break;
        case Activation::TANH:
            function = new TanH();
            break;
        default:
            function = new Linear();
    }
    this->function = function;
    this->seed = seed;
    this->instanceList = instanceList;
    hiddenLayers.insert(hiddenLayers.begin() + 0, instanceList.getInput());
    hiddenLayers.push_back(instanceList.getOutput());
    this->layers = (Layer *)calloc(hiddenLayers.size(), sizeof(Layer));
    this->layersLength = hiddenLayers.size();
    for (int i = 0; i < hiddenLayers.size(); i++) {
        if (i + 1 < hiddenLayers.size()) {
            this->layers[i] = Layer(hiddenLayers.at(i), hiddenLayers.at(i + 1), seed);
        } else {
            this->layers[i] = Layer(hiddenLayers.at(i));
        }
    }
    biases = (Bias *)calloc(hiddenLayers.size() - 1, sizeof(Bias));
    for (int i = 0; i < hiddenLayers.size() - 1; i++) {
        biases[i] = Bias(seed, layers[i + 1].getSize());
    }
}

vector<vector<double>> NeuralNetwork::calculateError(int i, vector<vector<vector<double>>>& deltaWeights) {
    deltaWeights[0] = multiplyMatrices(layers[i + 1].weightsToMatrix(), deltaWeights[0]);
    deltaWeights[0] = hadamardProduct(deltaWeights[0], function->calculateBack(layers[i + 1].neuronsToVector(), layers[i + 1].getSize()));
    return deltaWeights[0];
}

vector<vector<double>> NeuralNetwork::hadamardProduct(vector<vector<double>> matrix1, vector<vector<double>> matrix2) {
    vector<vector<double>> matrix = vector<vector<double>>();
    for (int i = 0; i < matrix1.size(); i++) {
        matrix.emplace_back();
        for (int j = 0; j < matrix1[0].size(); j++) {
            matrix[matrix.size() - 1].emplace_back();
        }
    }
    for (int i = 0; i < matrix1.size(); i++) {
        for (int j = 0; j < matrix1[i].size(); j++) {
            matrix[i][j] = matrix1[i][j] * matrix2[i][j];
        }
    }
    return matrix;
}

double NeuralNetwork::dotProduct(vector<double> vec, vector<vector<double>> matrix, int j) {
    double sum = 0.0;
    for (int i = 0; i < vec.size(); i++) {
        sum += vec[i] * matrix[i][j];
    }
    return sum;
}

vector<vector<double>> NeuralNetwork::multiplyMatrices(vector<vector<double>> matrix1, vector<vector<double>> matrix2) {
    vector<vector<double>> matrix = vector<vector<double>>();
    for (int i = 0; i < matrix1.size(); i++) {
        matrix.emplace_back();
        for (int j = 0; j < matrix2[0].size(); j++) {
            matrix[matrix.size() - 1].emplace_back();
        }
    }
    for (int j = 0; j < matrix2[0].size(); j++) {
        for (int i = 0; i < matrix1.size(); i++) {
            matrix[i][j] = dotProduct(matrix1[i], matrix2, j);
        }
    }
    return matrix;
}

void NeuralNetwork::calculateRMinusY(vector<vector<vector<double>>>& deltaWeights, int classInfo, double learningRate) {
    vector<vector<double>> matrix = vector<vector<double>>();
    for (int i = 0; i < layers[layersLength - 1].getSize(); i++) {
        matrix.emplace_back();
        matrix[matrix.size() - 1].emplace_back();
    }
    if (layers[layersLength - 1].getSize() > 1) {
        for (int j = 0; j < layers[layersLength - 1].getSize(); j++) {
            if (classInfo == j) {
                matrix[j][0] = learningRate * (1 - layers[layersLength - 1].getNeuron(j).getValue());
            } else {
                matrix[j][0] = learningRate * -layers[layersLength - 1].getNeuron(j).getValue();
            }
        }
    } else {
        matrix[0][0] = learningRate * (classInfo - layers[layersLength - 1].getNeuron(0).getValue());
    }
    deltaWeights.insert(deltaWeights.begin() + 0, matrix);
    deltaWeights.at(0) = multiplyMatrices(deltaWeights.at(0), layers[layersLength - 2].neuronsToMatrix());
    if (layersLength > 2) {
        deltaWeights.insert(deltaWeights.begin() + 0, matrix);
    }
}

void NeuralNetwork::setWeights(vector<vector<vector<double>>> deltaWeights, vector<vector<vector<double>>> oldDeltaWeights, double momentum) {
    for (int t = 0; t < deltaWeights.size(); t++) {
        vector<vector<double>> weights = deltaWeights[t];
        for (int i = 0; i < weights.size(); i++) {
            for (int j = 0; j < weights[i].size(); j++) {
                if (!oldDeltaWeights.empty()) {
                    weights[i][j] += (momentum * oldDeltaWeights[t][i][j]);
                }
                if (j > 0) {
                    layers[t].getNeuron(j - 1).addWeight(i, weights[i][j]);
                } else {
                    biases[t].addWeight(i, weights[i][j]);
                }
            }
        }
    }
}

vector<vector<vector<double>>> NeuralNetwork::backpropagation(int classInfo, double learningRate, double momentum, vector<vector<vector<double>>> oldDeltaWeights) {
    vector<vector<vector<double>>> deltaWeights = vector<vector<vector<double>>>();
    calculateRMinusY(deltaWeights, classInfo, learningRate);
    for (int i = layersLength - 3; i > -1; i--) {
        vector<vector<double>> currentError = calculateError(i, deltaWeights);
        deltaWeights.at(0) = multiplyMatrices(deltaWeights.at(0), layers[i].neuronsToMatrix());
        if (i > 0) {
            deltaWeights.insert(deltaWeights.begin() + 0, currentError);
        }
    }
    setWeights(deltaWeights, std::move(oldDeltaWeights), momentum);
    return deltaWeights;
}

void NeuralNetwork::train(int epoch, double learningRate, double etaDecrease, double momentum) {
    vector<vector<vector<double>>> oldDeltaWeights = vector<vector<vector<double>>>();
    for (int i = 0; i < epoch; i++) {
        instanceList.shuffle(seed);
        for (int j = 0; j < instanceList.size(); j++) {
            createInputVector(instanceList.getInstance(j));
            feedForward();
            oldDeltaWeights = backpropagation(instanceList.get(instanceList.getInstance(j).getLast()), learningRate, momentum, oldDeltaWeights);
        }
        learningRate *= etaDecrease;
    }
}

void NeuralNetwork::createInputVector(Instance instance) {
    for (int i = 0; i < layers[0].getSize(); i++) {
        layers[0].getNeuron(i).setValue(std::stoi(instance.get(i)));
    }
}

void NeuralNetwork::feedForward() {
    for (int i = 0; i < layersLength - 1; i++) {
        for (int j = 0; j < layers[i + 1].getSize(); j++) {
            double sum = 0.0;
            for (int k = 0; k < layers[i].getSize(); k++) {
                sum += layers[i].getNeuron(k).getWeight(j) * layers[i].getNeuron(k).getValue();
            }
            sum += biases[i].getValue(j);
            if (i + 1 != layersLength - 1) {
                sum = function->calculateForward(sum);
            }
            layers[i + 1].getNeuron(j).setValue(sum);
        }
    }
    if (layers[layersLength - 1].getSize() > 2) {
        layers[layersLength - 1].softmax();
    }
}

string NeuralNetwork::predict(Instance instance) {
    createInputVector(std::move(instance));
    feedForward();
    if (instanceList.getOutput() == 1) {
        double outputValue = layers[layersLength - 1].getNeuron(0).getValue();
        if (outputValue >= 0.5) {
            return instanceList.get(1);
        }
        return instanceList.get(0);
    }
    double bestValue = INT32_MIN;
    int bestNeuron = -1;
    for (int i = 0; i < layers[layersLength - 1].getSize(); i++) {
        if (layers[layersLength - 1].getNeuron(i).getValue() > bestValue) {
            bestValue = layers[layersLength - 1].getNeuron(i).getValue();
            bestNeuron = i;
        }
    }
    return instanceList.get(bestNeuron);
}

double NeuralNetwork::test(InstanceList list) {
    int correct = 0;
    int total = 0;
    for (int i = 0; i < list.size(); i++) {
        Instance instance = list.getInstance(i);
        if (instance.getLast() == predict(instance)) {
            correct++;
        }
        total++;
    }
    return correct * 100.00 / total;
}

#endif //NEURALNETWORKS_CPP_NEURALNETWORK_H
