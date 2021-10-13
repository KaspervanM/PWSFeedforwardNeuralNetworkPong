// Model.h - Contains declarations of Model
#pragma once

#ifdef NNMODEL_EXPORTS
#define NNMODEL_API __declspec(dllexport)
#else
#define NNMODEL_API __declspec(dllimport)
#endif

#include <vector>

enum class act {
	identity,
	binary,
	sigmoid,
	tanh,
	ReLU,
	leaky_ReLU,
	ReLU6,
	leaky_ReLU6,
	softplus,
	softsign,
	softmax,
	gaussian,
	swish,
	hard_swish,
	fastSigmoid
};

struct ModelArgs {
	//int iterations = 0;
	int seed = -1;
	std::vector<int> shape;
	std::vector<act> actFuncts;

	ModelArgs& operator=(const ModelArgs& a)
	{
		if (this == &a)
			return *this; //self assignment

		//iterations = (a.iterations > 0) * a.iterations + !(a.iterations > 0) * iterations;
		seed = (a.seed > -1) * a.seed + !(a.seed > -1) * seed;
		shape = (a.shape.size() ? a.shape : shape);
		actFuncts = (a.actFuncts.size() ? a.actFuncts : actFuncts);

		return *this;
	}
};

class Model {
public:
	Model(ModelArgs modelArgs);
	Model(const Model& rhs);
	Model& operator=(const Model& rhs);
	void setWeightsBiases(const int seed);
	static void mutateModel(Model* model, const int seed, const double rate, const double degree);

	std::vector<std::vector<std::vector<double>>> weights;
	std::vector<std::vector<double>> biases;
	std::vector<int> shape;
	std::vector<act> actFuncts;
};

extern "C" NNMODEL_API Model * Model_new(const int seed, const unsigned int size, const unsigned int* shape, const unsigned int* actFuncts);
extern "C" NNMODEL_API int test();
extern "C" NNMODEL_API Model * Model_getMutated(Model * model, const int seed, const double rate, const double degree);
extern "C" NNMODEL_API double* Model_getWeights(Model * model, unsigned int& size);
extern "C" NNMODEL_API double* Model_getBiases(Model * model, unsigned int& size);
extern "C" NNMODEL_API unsigned int* Model_getShape(Model * model, unsigned int& size);
extern "C" NNMODEL_API unsigned int* Model_getActFuncts(Model * model, unsigned int& size);