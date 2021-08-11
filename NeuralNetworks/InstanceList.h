//
// Created by Oğuz Kerem Yıldız on 10.08.2021.
//

#ifndef NEURALNETWORKS_CPP_INSTANCELIST_H
#define NEURALNETWORKS_CPP_INSTANCELIST_H
#include "Instance.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>

using namespace std;

class InstanceList {
private:
    vector<Instance> list;
    unordered_map<string, int> classes;
    unordered_map<int, string> reverseClasses;
    int input = -1;
    static vector<string> split(const string& s, const string& regex);
public:
    InstanceList();
    InstanceList(const string& fileName, const string& regex);
    Instance getInstance(int i);
    void shuffle(int seed);
    int size();
    int getInput() const;
    int getOutput();
    int get(const string& key);
    string get(int neuron);
};

InstanceList::InstanceList() = default;

InstanceList::InstanceList(const string& fileName, const string& regex) {
    this->list = vector<Instance>();
    this->classes = unordered_map<string, int>();
    this->reverseClasses = unordered_map<int, string>();
    int currentKey = -1;
    fstream file;
    file.open(fileName, ios::in);
    if (file.fail()) {
        cout << "file not reading" << endl;
    } else {
        for (string current; getline(file, current);) {
            vector<string> line = split(current, regex);
            if (input < 0) {
                input = line.size() - 1;
            }
            string classInfo = line.at(line.size() - 1);
            if (classes.find(classInfo) == classes.end()) {
                currentKey++;
                classes[classInfo] = currentKey;
            }
            if (reverseClasses.find(currentKey) == reverseClasses.end()) {
                reverseClasses[currentKey] = classInfo;
            }
            list.emplace_back();
            for (const string& s : line) {
                list.at(list.size() - 1).add(s);
            }
        }
    }
}

vector<string> InstanceList::split(const string& s, const string& regex) {
    vector<string> vec = vector<string>();
    string current;
    for (int i = 0; i < s.size(); i++) {
        if (s.substr(i, regex.size()) == regex) {
            vec.push_back(current);
            current = "";
            i += regex.size() - 1;
        } else {
            current += s.at(i);
        }
    }
    if (!current.empty()) {
        vec.push_back(current);
    }
    return vec;
}

Instance InstanceList::getInstance(int i) {
    return this->list.at(i);
}

void InstanceList::shuffle(int seed) {
    std::default_random_engine e(seed);
    std::shuffle(std::begin(list), std::end(list), e);
}

int InstanceList::size() {
    return this->list.size();
}

int InstanceList::getInput() const {
    return this->input;
}

int InstanceList::getOutput() {
    if (classes.size() == 2) {
        return 1;
    }
    return classes.size();
}

int InstanceList::get(const string& key) {
    return this->classes[key];
}

string InstanceList::get(int neuron) {
    return this->reverseClasses[neuron];
}

#endif //NEURALNETWORKS_CPP_INSTANCELIST_H
