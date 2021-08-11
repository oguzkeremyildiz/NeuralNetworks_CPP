//
// Created by Oğuz Kerem Yıldız on 10.08.2021.
//

#ifndef NEURALNETWORKS_CPP_LINEAR_H
#define NEURALNETWORKS_CPP_LINEAR_H
#include "ActivationFunction.h"

using namespace std;

class Linear : public ActivationFunction {
public:
    double calculateForward(double value) override {
        return value;
    }
    vector<vector<double>> calculateBack(double* values, int valuesSize) override {
        vector<vector<double>> vec = vector<vector<double>>();
        for (int i = 0; i < valuesSize; i++) {
            vec.emplace_back();
            vec[vec.size() - 1].emplace_back();
        }
        for (int i = 0; i < valuesSize; i++) {
            vec[i][0] = 1.0;
        }
        return vec;
    }
    string to_string() override {
        return "LINEAR";
    }
};

#endif //NEURALNETWORKS_CPP_LINEAR_H
