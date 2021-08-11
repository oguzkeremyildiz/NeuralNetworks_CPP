//
// Created by Oğuz Kerem Yıldız on 11.08.2021.
//

#ifndef NEURALNETWORKS_CPP_TANH_H
#define NEURALNETWORKS_CPP_TANH_H
#include "ActivationFunction.h"

using namespace std;

class TanH : public ActivationFunction {
public:
    double calculateForward(double value) override {
        return (2.0 / (1.0 + std::exp(-2 * value))) - 1.0;
    }
    vector<vector<double>> calculateBack(double* values, int valuesSize) override {
        vector<vector<double>> matrix = vector<vector<double>>();
        for (int i = 0; i < valuesSize; i++) {
            matrix.emplace_back();
            matrix[matrix.size() - 1].emplace_back();
        }
        for (int i = 0; i < valuesSize; i++) {
            matrix[i][0] = 1.0 - (values[i] * values[i]);
        }
        return matrix;
    }
    string to_string() override {
        return "TANH";
    }
};

#endif //NEURALNETWORKS_CPP_TANH_H
