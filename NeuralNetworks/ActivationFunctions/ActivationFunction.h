//
// Created by Oğuz Kerem Yıldız on 10.08.2021.
//

#ifndef NEURALNETWORKS_CPP_ACTIVATIONFUNCTION_H
#define NEURALNETWORKS_CPP_ACTIVATIONFUNCTION_H
#include <string>

using namespace std;

class ActivationFunction {
public:
    virtual ~ActivationFunction() {}
    virtual double calculateForward(double value) = 0;
    virtual vector<vector<double>> calculateBack(double* values, int valuesSize) = 0;
    virtual string to_string() = 0;
};
#endif //NEURALNETWORKS_CPP_ACTIVATIONFUNCTION_H
