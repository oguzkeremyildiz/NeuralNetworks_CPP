//
// Created by Oğuz Kerem Yıldız on 8.08.2021.
//

#ifndef NEURALNETWORKS_CPP_NEURON_H
#define NEURALNETWORKS_CPP_NEURON_H
#include <string>

using namespace std;

class Neuron {
private:
    double value;
    double* weights;
    int size;
public:
    Neuron();
    Neuron(double* weights, int size);
    double getWeight(int i);
    void setWeight(int i, double weight);
    void addWeight(int i, double weight);
    double getValue();
    void setValue(double val);
    void initializeWeight(int size);
    string to_string();
    int getSize();
};

Neuron::Neuron() {
    this->value = 0.0;
    this->weights = nullptr;
    this->size = 0;
}

Neuron::Neuron(double *weights, int size) {
    this->value = 0.0;
    this->weights = weights;
    this->size = size;
}

int Neuron::getSize() {
    return this->size;
}

double Neuron::getWeight(int i) {
    return this->weights[i];
}

void Neuron::setWeight(int i, double weight) {
    this->weights[i] = weight;
}

void Neuron::addWeight(int i, double weight) {
    this->weights[i] += weight;
}

double Neuron::getValue() {
    return this->value;
}

void Neuron::setValue(double val) {
    this->value = val;
}

void Neuron::initializeWeight(int size) {
    this->size = size;
    this->weights = (double *)calloc(size, sizeof(double));
}

string Neuron::to_string() {
    string sb;
    sb += std::to_string(value) + " ";
    if (weights != nullptr) {
        for (int i = 0; i < size; i++) {
            sb += std::to_string(weights[i]) + " ";
        }
    }
    return sb;
}

#endif //NEURALNETWORKS_CPP_NEURON_H
