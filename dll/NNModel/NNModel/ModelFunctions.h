// ModelFunctions.h - Contains declarations of model functions
#pragma once

#include "Model.h"
#include "TrainingData.h"

//#define max(A,B) (A >= B ? A : B)
//#define min(A,B) (A <= B ? A : B)
#define ReLU6_6 6
#define leaky_ReLU_slope .1

class ModelFunctions
{
public:
	template<typename T>
	static std::vector<std::vector<T>> matmul(std::vector<std::vector<T>> x, std::vector<T> y);
	template<typename T>
	static std::vector<T> flatten(std::vector<std::vector<T>> x);
	template<typename T>
	static double Exp(T x);
	template<typename T>
	static std::vector<T> Exp(std::vector<T> x);
	template<typename T>
	static double activation(T z, act type);
	template<typename T>
	static std::vector<T> activation(std::vector<T> z, act type);
	template<typename T>
	static double activation_derivative(T z, act type);
	template<typename T>
	static std::vector<T> activation_derivative(std::vector<T> z, act type);
	static std::vector<double> feedforward(Model* model, std::vector<double> inp);
	static std::vector<std::vector<double>> getNodevals(Model* model, std::vector<double> inp);
	static double cost(Model* model, std::vector<double> output, std::vector<double> desiredOutput, const int i);
	static double cost_derivative(Model* model, std::vector<double> output, std::vector<double> desiredOutput, const int j, const int i);
	static std::vector<std::vector<double>> dCdl(Model* model, std::vector<std::vector<double>> nodevals, std::vector<double> desiredOutput, int curr_layer, std::vector<std::vector<double>> ret);
	static void backpropagation(Model* model, std::vector<std::vector<double>> nodevals, std::vector<double> desiredOutput, std::vector<std::vector<std::vector<double>>>& batch_dCdw_i, std::vector<std::vector<double>>& batch_dCdb_i);
	static void optimize(Model* model, const double learning_rate, std::vector<std::vector<std::vector<std::vector<double>>>> batch_dCdw, std::vector<std::vector<std::vector<double>>> batch_dCdb);
	static void train(Model* model, const int iterations, TrainingData* TD, const double learning_rate);
};

extern "C" NNMODEL_API double* feedforward(Model * model, double* input);
extern "C" NNMODEL_API void freeOutD(double* out);
extern "C" NNMODEL_API void freeOutI(unsigned int* out);
extern "C" NNMODEL_API void freeOutM(Model * out);
extern "C" NNMODEL_API void trainFromFile(Model * model,
	const unsigned int iterations,
	const double learning_rate,
	const char* path_to_file,
	const unsigned int batch_size,
	const double percentage_of_data_not_to_train,
	const int training_data_seed
);
extern "C" NNMODEL_API void trainFromArray(Model * model,
	const unsigned int iterations,
	const double learning_rate,
	const double* trainingDataArray,
	const unsigned int trainingDataSamples,
	const unsigned int batch_size,
	const double percentage_of_data_not_to_train,
	const int training_data_seed
);
