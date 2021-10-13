// Model.cpp : Defines the exported functions for the DLL.
#include "pch.h"
#include "Model.h"
#include "ShowVector.h"
#include <algorithm>
#include <random>

Model::Model(ModelArgs modelArgs)
	: shape(modelArgs.shape),
	actFuncts(modelArgs.actFuncts)
{
	// Set the weights
	setWeightsBiases(modelArgs.seed);
}

Model::Model(const Model& rhs) {
	weights = rhs.weights;
	biases = rhs.biases;
	shape = rhs.shape;
	actFuncts = rhs.actFuncts;
}

Model& Model::operator=(const Model& rhs) {
	if (this == &rhs) {
		return *this;
	}
	weights = rhs.weights;
	biases = rhs.biases;
	shape = rhs.shape;
	actFuncts = rhs.actFuncts;
	return *this;
}

void Model::setWeightsBiases(const int seed)
{
	// Initialize a random generator.
	std::mt19937 engine;

	// Set the generator with a seed. A random seed will be chose if the seed is below zero.
	if (seed < 0) engine = std::mt19937(std::random_device{}());
	else engine = std::mt19937(seed);

	// Create the distribution object for generating random doubles between -1 and 1.
	// A uniform distribution is used.
	std::uniform_real_distribution<double> rand_double(-1.0, 1.0);

	auto gen = [&rand_double, &engine]() {
		return rand_double(engine);
	};

	// Then, the weights are defined and set.
	weights.resize(shape.size() - 1);
	for (size_t i = 0; i < shape.size() - 1; i++) {
		weights[i].resize(shape[i + 1]);
		for (int j = 0; j < shape[i + 1]; j++) {
			weights[i][j].resize(shape[i]);
			std::generate(std::begin(weights[i][j]), std::end(weights[i][j]), gen);
		}
	}

	// The same goes for the biases.
	biases.resize(shape.size() - 1);
	for (size_t i = 0; i < shape.size() - 1; i++) {
		biases[i].resize(shape[i + 1]);
		std::generate(std::begin(biases[i]), std::end(biases[i]), gen);
	}
}
void Model::mutateModel(Model* model, const int seed, const double rate, const double degree)
{
	// Initialize a random generator.
	std::mt19937 engine;

	// Set the generator with a seed. A random seed will be chose if the seed is below zero.
	if (seed < 0) engine = std::mt19937(std::random_device{}());
	else engine = std::mt19937(seed);

	// Create the distribution object for generating random doubles between -1 and 1.
	// A uniform distribution is used.
	std::normal_distribution<double> rand_double(0, degree);
	std::uniform_real_distribution<double> urand_double(-1.0, 1.0);

	// The weights are mutated.
	for (auto& l : model->weights) {
		for (auto& n : l) {
			for (auto& w : n) {
				if (urand_double(engine) / 2 + .5 < rate) {
					w += rand_double(engine);
				}
			}
		}
	}

	// The same goes for the biases.
	for (auto& l : model->biases) {
		for (auto& b : l) {
			if (urand_double(engine) / 2 + .5 < rate) {
				b += rand_double(engine);
			}
		}
	}
}

bool modelArgsChecker(ModelArgs modelArgs)
{
	if (std::any_of(modelArgs.shape.begin(), modelArgs.shape.end(), [](int i) {return i <= 0; }) || modelArgs.shape.size() <= 1) {
		return false;
	}
	if (modelArgs.shape.size() != modelArgs.actFuncts.size() + 1) {
		return false;
	}
	return true;
}

Model* Model_new(const int seed, const unsigned int size, const unsigned int* shape, const unsigned int* actFuncts)
{
	ModelArgs modelArgs;
	modelArgs.seed = seed;

	std::vector<int> _shape;
	for (size_t i = 0; i < size; i++)
		_shape.push_back(shape[i]);
	modelArgs.shape = _shape;

	std::vector<act> _actFuncts;
	for (size_t i = 0; i < size - 1; i++)
		_actFuncts.push_back((act)actFuncts[i]);
	modelArgs.actFuncts = _actFuncts;

	if (modelArgsChecker(modelArgs))
		return new Model(modelArgs);
	return nullptr;
}

int test() {
	return 5;
}

Model* Model_getMutated(Model* model, const int seed, const double rate, const double degree)
{
	Model* newModel = new Model(*model);
	Model::mutateModel(newModel, seed, rate, degree);
	return newModel;
}

double* Model_getWeights(Model* model, unsigned int& size)
{
	size = 0;
	for (auto& l : model->weights) {
		for (auto& n : l) {
			size += n.size();
		}
	}
	double* weights = new double[size];

	unsigned int i = 0;
	for (auto& l : model->weights) {
		for (auto& n : l) {
			for (auto& w : n) {
				weights[i++] = w;
			}
		}
	}

	return weights;
}

double* Model_getBiases(Model* model, unsigned int& size)
{
	size = 0;
	for (auto& l : model->biases)
		size += l.size();
	double* biases = new double[size];

	unsigned int i = 0;
	for (auto& l : model->biases)
		for (auto& b : l)
			biases[i++] = b;

	return biases;
}

unsigned int* Model_getShape(Model* model, unsigned int& size)
{
	size = model->shape.size();
	unsigned int* shape = new unsigned int[size];

	unsigned int i = 0;
	for (auto& l : model->shape)
		shape[i++] = l;

	return shape;
}

unsigned int* Model_getActFuncts(Model* model, unsigned int& size)
{
	size = model->actFuncts.size();
	unsigned int* actFuncts = new unsigned int[size + 1];

	unsigned int i = 0;
	for (auto& f : model->actFuncts)
		actFuncts[i++] = (unsigned int)f;

	return actFuncts;
}