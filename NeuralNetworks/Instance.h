//
// Created by Oğuz Kerem Yıldız on 10.08.2021.
//

#ifndef NEURALNETWORKS_CPP_INSTANCE_H
#define NEURALNETWORKS_CPP_INSTANCE_H
#include <utility>
#include <vector>
#include <string>

using namespace std;

class Instance {
private:
    vector<string> list;
public:
    Instance();
    explicit Instance(vector<string> vec);
    void add(string s);
    string get(int index);
    string getLast();
};

Instance::Instance() {
    this->list = vector<string>();
}

Instance::Instance(vector<string> vec) {
    this->list = std::move(vec);
}

void Instance::add(string s) {
    this->list.push_back(s);
}

string Instance::get(int index) {
    return this->list.at(index);
}

string Instance::getLast() {
    return this->list.at(list.size() - 1);
}

#endif //NEURALNETWORKS_CPP_INSTANCE_H
