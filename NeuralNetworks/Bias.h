//
// Created by Oğuz Kerem Yıldız on 10.08.2021.
//

#ifndef NEURALNETWORKS_CPP_BIAS_H
#define NEURALNETWORKS_CPP_BIAS_H
#include <random>
#include <string>

using namespace std;

class Bias {
private:
    double* weights;
    int size;
public:
    Bias(int size);
    Bias(int seed, int size);
    double getValue(int i);
    void setWeight(int index, double weight);
    void addWeight(int index, double weight);
    string to_string();
};

Bias::Bias(int size) {
    this->size = size;
    this->weights = (double *)calloc(size, sizeof(double));
}

Bias::Bias(int seed, int size) {
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    this->size = size;
    weights = (double *)calloc(size, sizeof(double));
    for (int i = 0; i < size; i++) {
        weights[i] = 2 * distribution(generator) - 1;
    }
}

double Bias::getValue(int i) {
    return this->weights[i];
}

void Bias::setWeight(int index, double weight) {
    this->weights[index] = weight;
}

void Bias::addWeight(int index, double weight) {
    this->weights[index] += weight;
}

string Bias::to_string() {
    string sb;
    for (int i = 0; i < size; i++) {
        sb += std::to_string(weights[i]) + " ";
    }
    return sb;
}

#endif //NEURALNETWORKS_CPP_BIAS_H
