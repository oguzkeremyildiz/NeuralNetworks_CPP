//
// Created by Oğuz Kerem Yıldız on 10.08.2021.
//
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../NeuralNetworks/NeuralNetwork.h"
#include <vector>
using namespace std;

TEST_CASE("NeuralNetworkTest-test") {
    InstanceList list = InstanceList("dermatology.txt", ",");
    vector<int> layers = vector<int>();
    layers.push_back(20);
    NeuralNetwork net = NeuralNetwork(1, layers, list, Activation::SIGMOID);
    net.train(100, 0.01, 0.99, 0.5);
    REQUIRE_THAT(98.6338797814, Catch::Matchers::WithinAbs(net.test(list), 0.01));
}
