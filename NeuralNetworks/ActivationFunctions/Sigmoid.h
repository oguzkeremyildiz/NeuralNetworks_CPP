//
// Created by Oğuz Kerem Yıldız on 10.08.2021.
//

#ifndef NEURALNETWORKS_CPP_SIGMOID_H
#define NEURALNETWORKS_CPP_SIGMOID_H
#include "ActivationFunction.h"

using namespace std;

class Sigmoid : public ActivationFunction {
private:
    static vector<vector<double>> multiply(const double* values, const double* oneMinusValues, int valuesSize) {
        vector<vector<double>> matrix = vector<vector<double>>();
        for (int i = 0; i < valuesSize; i++) {
            matrix.emplace_back();
            matrix[matrix.size() - 1].emplace_back();
        }
        for (int i = 0; i < valuesSize; i++) {
            matrix[i][0] = values[i] * oneMinusValues[i];
        }
        return matrix;
    }
public:
    double calculateForward(double value) override {
        return 1.0 / (1.0 + std::exp(-value));
    }
    vector<vector<double>> calculateBack(double* values, int valuesSize) override {
        auto* oneMinusValues = (double *)calloc(valuesSize, sizeof(double));
        for (int i = 0; i < valuesSize; i++) {
            oneMinusValues[i] = 1.0 - values[i];
        }
        return multiply(values, oneMinusValues, valuesSize);
    }
    string to_string() override {
        return "SIGMOID";
    }
};
#endif //NEURALNETWORKS_CPP_SIGMOID_H
