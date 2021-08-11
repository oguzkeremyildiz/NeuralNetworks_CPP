//
// Created by Oğuz Kerem Yıldız on 11.08.2021.
//

#ifndef NEURALNETWORKS_CPP_RELU_H
#define NEURALNETWORKS_CPP_RELU_H
#include "ActivationFunction.h"
#include <algorithm>

using namespace std;

class ReLU : public ActivationFunction {
public:
    double calculateForward(double value) override {
        return max(0.0, value);
    }
    vector<vector<double>> calculateBack(double* values, int valuesSize) override {
        vector<vector<double>> vec = vector<vector<double>>();
        for (int i = 0; i < valuesSize; i++) {
            vec.emplace_back();
            vec[vec.size() - 1].emplace_back();
        }
        for (int i = 0; i < valuesSize; i++) {
            vec[i][0] = 0.0;
            if (values[i] > 0) {
                vec[i][0] = 1.0;
            }
        }
        return vec;
    }
    string to_string() override {
        return "RELU";
    }
};
#endif //NEURALNETWORKS_CPP_RELU_H
