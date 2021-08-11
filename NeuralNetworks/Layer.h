//
// Created by Oğuz Kerem Yıldız on 8.08.2021.
//

#ifndef NEURALNETWORKS_CPP_LAYER_H
#define NEURALNETWORKS_CPP_LAYER_H
#include "Neuron.h"
#include <random>

using namespace std;

class Layer {
private:
    Neuron* neurons;
    int size;
    int nextSize;
public:
    Layer(int size);
    Layer(int size, int nextSize, int seed);
    Neuron& getNeuron(int i);
    void softmax();
    int getSize() const;
    vector<vector<double>> neuronsToMatrix();
    double* neuronsToVector();
    vector<vector<double>> weightsToMatrix();
};

Layer::Layer(int size) {
    this->size = size;
    this->nextSize = 0;
    this->neurons = (Neuron *)calloc(size, sizeof(Neuron));
    for (int i = 0; i < size; i++) {
        neurons[i] = Neuron();
    }
}

Layer::Layer(int size, int nextSize, int seed) {
    this->size = size;
    this->nextSize = nextSize;
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    this->neurons = (Neuron *)calloc(size, sizeof(Neuron));
    for (int i = 0; i < size; i++) {
        auto* weights = (double *)calloc(nextSize, sizeof(double ));
        for (int j = 0; j < nextSize; j++) {
            weights[j] = 2 * distribution(generator) - 1;
        }
        neurons[i] = Neuron(weights, nextSize);
    }
}

Neuron& Layer::getNeuron(int i) {
    return this->neurons[i];
}

void Layer::softmax() {
    double total = 0.0;
    for (int i = 0; i < size; i++) {
        total += std::exp(getNeuron(i).getValue());
    }
    for (int i = 0; i < size; i++) {
        double value = getNeuron(i).getValue();
        getNeuron(i).setValue(std::exp(value) / total);
    }
}

int Layer::getSize() const {
    return this->size;
}

vector<vector<double>> Layer::neuronsToMatrix() {
    vector<vector<double>> neurons = vector<vector<double>>();
    neurons.emplace_back();
    for (int i = 0; i < size + 1; i++) {
        neurons[0].emplace_back();
    }
    neurons[0][0] = 1.0;
    for (int i = 0; i < size; i++) {
        neurons[0][i + 1] = this->neurons[i].getValue();
    }
    return neurons;
}

double *Layer::neuronsToVector() {
    auto* neurons = (double *)calloc(size, sizeof(double));
    for (int i = 0; i < size; i++) {
        neurons[i] = this->neurons[i].getValue();
    }
    return neurons;
}

vector<vector<double>> Layer::weightsToMatrix() {
    vector<vector<double>> weights = vector<vector<double>>();
    for (int i = 0; i < size; i++) {
        weights.emplace_back();
        for (int j = 0; j < nextSize; j++) {
            weights[weights.size() - 1].emplace_back();
        }
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < nextSize; j++) {
            weights[i][j] = this->neurons[i].getWeight(j);
        }
    }
    return weights;
}

#endif //NEURALNETWORKS_CPP_LAYER_H
