// ModelFunctions.cpp : Defines the exported functions for the DLL.
#include "pch.h"
#include "ModelFunctions.h"
#include "ShowVector.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

// This is the function that deals with matrix multiplications and dot products. It function
// contains a form of error handling, because it would work unpredictably when shapes do not match
// up. @param x is a a 2d or 1d array. @param y is a 1d array, but if it contains only one value, it
// is considered a scalar. @param shape determines the shape of x and whether it is a 1d or 2d
// array. @param size1 determines the size of x. @param size2 determines the size of y.
template<typename T>
std::vector<std::vector<T>> ModelFunctions::matmul(std::vector<std::vector<T>> x, std::vector<T> y)
{
	const size_t xheight = x.size();
	const size_t ysize = y.size();
	const size_t xwidth = x[0].size();

	// Error handling.
	if (xheight == 0 || ysize == 0) {
		printf("ERROR AT matmul(): PARAMETER OF SIZE 0 WAS GIVEN.");
		return std::vector<std::vector<T>>();
	}
	if (xwidth != ysize && ysize != 1 && xheight != ysize && xwidth != 1) {
		printf("ERROR AT matmul(): SIZE OF VECOTER IN FIRST PARAMETER DOES NOT MATCH SIZE OF SECOND PARAMETER OR SIZE OF PARAMETER DOES NOT MATCH SIZE OF SECOND PARAMETER OR SECOND PARAMETER IS NOT SCALAR.");
		return std::vector<std::vector<T>>();
	}
	for (auto& i : x) {
		if (i.size() != xwidth) {
			printf("ERROR AT matmul(): SIZE OF VECTORS IN FIRST PARAMETER ARE NOT EQUAL.");
			return std::vector<std::vector<T>>();
		}
	}

	// If y contains only one value, it should be considered a scalar. If the second dimention of x
	// only contains single values, @param x should be considered a column vector. Else it should
	// just perform a matrix multiplication.
	if (ysize == 1) {
		// x will contain the product of x and y
		for (auto& i : x)
			std::transform(i.begin(), i.end(), i.begin(), [&y](auto j) -> T {return j * y[0]; });
		return(x);
	}
	else if (xwidth == 1) {
		// x will contain the product of x and y
		for (auto& i : x) {
			i.resize(ysize);
			const T c = i[0];
			std::transform(i.begin(), i.end(), i.begin(), [&y, &c, idx = -1](auto j) mutable {
				return y[++idx] * c;
			});
		}
		return x;
	}
	else {
		// xy will contain the product of x and y
		std::vector<std::vector<T>> xy(xheight, std::vector<T>(1));
		for (int i = 0; i < xheight; i++) {
			xy[i][0] = (T)std::inner_product(x[i].begin(), x[i].end(), y.begin(), 0.);
		}
		return xy;
	}
}

template<typename T>
std::vector<T> ModelFunctions::flatten(std::vector<std::vector<T>> x)
{
	std::vector<T> ret;
	for (const auto& v : x)
		ret.insert(ret.end(), v.begin(), v.end());
	return ret;
}

template<typename Real, size_t degree, size_t i = 0>
struct Recursion {
	static Real evaluate(Real x) {
		constexpr Real c = 1.0 / static_cast<Real>(1u << degree);
		x = Recursion<Real, degree, i + 1>::evaluate(x);
		return x * x;
	}
};

template<typename Real, size_t degree>
struct Recursion<Real, degree, degree> {
	static Real evaluate(Real x) {
		constexpr Real c = 1.0 / static_cast<Real>(1u << degree);
		x = 1.0 + c * x;
		return x;
	}
};

// ---------Timesave by lowering the degree (10)-----------
template<typename T>
double ModelFunctions::Exp(T x) {
	double ret = Recursion<double, 15>::evaluate(x);
	return ret;
}
template<typename T>
std::vector<T> ModelFunctions::Exp(std::vector<T> x) {
	for (auto& i : x)
		i = Exp(i);
	return x;
}

// This function contains the activation functions. @param size contains the size of z. @param z
// contains the array that will be put into the activation function. @param i determines which
// function to use. By default, the fast sigmoid is used.
template<typename T>
double ModelFunctions::activation(T z, act type)
{
	switch (type)
	{
	case act::identity: // Identity
		return z;
	case act::binary: // Binary step
		return (z <= 0 ? 0 : 1);
	case act::sigmoid: // Sigmoid function.
		return (1 / (1 + Exp(-z)));
	case act::tanh: // tanh
		return (1 - 2 / (1 + Exp(2 * z)));
	case act::ReLU: // ReLU
		return max(z, 0);
	case act::leaky_ReLU:
		return max(leaky_ReLU_slope * z, 0);
	case act::ReLU6:
		return min(max(z, 0), ReLU6_6);
	case act::leaky_ReLU6:
		return min(max(leaky_ReLU_slope * z, 0), 6);
	case act::softplus: // Softplus
		return std::log(1 + Exp(z));
	case act::softsign: // Softsign
		return (z / (1 + abs(z)));
	case act::gaussian: // Gaussian
		return Exp(-1 * (z * z));
	case act::swish:
		return (z * activation(z, act::sigmoid));
	case act::hard_swish:
		return (z * activation(z + ReLU6_6 / 2, act::ReLU6) / ReLU6_6);
	default: // 'Fast sigmoid': an approximation of the sigmoid function, but a bit faster.
		return (.5 + z * .5 / (1 + abs(z)));
	}
}

template<typename T>
std::vector<T> ModelFunctions::activation(std::vector<T> z, act type)
{
	if (type == act::softmax) { // Softmax
		double sum = 0;
		for (auto i : z)
			sum += Exp(i);

		for (auto& i : z)
			i = Exp(i) / sum;
		return z;
	}
	for (auto& i : z)
		i = activation(i, type);
	return z;
}

// This function contains the derivative of the activation functions. @param size contains the size
// of z. @param z contains the array that will be put into the derivative of the activation
// function. @param type determines which function to use. By default, the fast sigmoid is used.
template<typename T>
double ModelFunctions::activation_derivative(T z, act type)
{
	switch (type)
	{
	case act::identity: // Identity
		return 1;
	case act::binary: // Binary step
		return 0;
	case act::sigmoid: // Sigmoid function.
	{
		double act = activation(z, type);
		return (act * (1.0 - act));
	}
	case act::tanh: // tanh
	{
		double act = activation(z, type);
		return (1. - act * act);
	}
	case act::ReLU: // ReLU
		return (z <= 0 ? 0 : 1);
	case act::leaky_ReLU:
		return (z <= 0 ? leaky_ReLU_slope : 1);
	case act::ReLU6:
		return (z <= 0 ? 0 : (z <= ReLU6_6 ? 1 : 0));
	case act::leaky_ReLU6:
		return (z <= 0 ? leaky_ReLU_slope : (z <= ReLU6_6 ? 1 : 0));
	case act::softplus: // Softplus
		return activation(z, act::sigmoid);;
	case act::softsign: // Softsign
		return (1 / ((1 + abs(z)) * (1 + abs(z))));
	case act::gaussian: // Gaussian
		return (2 * z * Exp(-1 * (z * z)));
	case act::swish:             // ------------------ TODO -------------------
		return z * activation(z, act::sigmoid);
	case act::hard_swish:        // ------------------ TODO -------------------
		return z * activation(z + ReLU6_6 / 2, act::ReLU6) / ReLU6_6;
	default: // 'Fast sigmoid': an approximation of the sigmoid function, but a bit faster.
		return .5 / (1 + z * z + 2 * abs(z));
	}
}

template<typename T>
std::vector<T> ModelFunctions::activation_derivative(std::vector<T> z, act type)
{
	if (type == act::softmax) { // Softmax
		std::vector<T> act = activation(z, type);
		for (int i = 0; i < (int)z.size(); i++) {
			z[i] = act[i] * (1.0 - act[i]);
		}
		return z;
	}
	for (auto& i : z) {
		i = activation_derivative(i, type);
	}
	return z;
}

// This function performs the feedforward process and gets all the node values. @param model is the
// model on which the operation will be performed. @param inp is the input layer. @param nodevals
// will contain all node values. @return The output is a vector containing all the node values. It
// loops through all layers calculating the values of each node and storing them in model->nodevals.
// This will be used again in other functions.
std::vector<std::vector<double>> ModelFunctions::getNodevals(Model* model, std::vector<double> inp)
{
	// Check if the input array has the correct size.
	if (inp.size() != model->shape[0]) {
		printf("Input size: %d\nExpected: %d", (int)inp.size(), model->shape[0]);
		return std::vector<std::vector<double>>();
	}

	// Empty the vector that will hold all the node values.
	std::vector<std::vector<double>> nodevals;

	// Set the input layer.
	nodevals.push_back(inp);

	// Loop through all layers starting from layer 2.
	const size_t size_1 = model->shape.size() - 1;
	for (int i = 0; i < size_1; i++) {
		std::vector<double> z = flatten(matmul(model->weights[i], nodevals[i]));
		int c = 0;
		for (auto& v : z) {
			v += model->biases[i][c++];
		}
		nodevals.push_back(activation(z, model->actFuncts[i]));
	}
	return nodevals;
}

// This function performs the feedforward process. @param model is the model on which the operation
// will be performed. @param inp is the input layer. @return output is the output layer. It loops
// through all layers calculating the values of each node and storing them in model->nodevals. This
// will be used again in other functions.
std::vector<double> ModelFunctions::feedforward(Model* model, std::vector<double> inp)
{
	return getNodevals(model, inp).back();
}

// This function contains all the cost functions (now only the mean squared error function).
double ModelFunctions::cost(Model* model, std::vector<double> output, std::vector<double> desiredOutput, const int i)
{
	if (desiredOutput.size() == 0) {
		printf("COST: Derired input was not set");
		return NULL;
	}
	switch (i)
	{
	case 1:
	{
		// TODO: Add more cost functions
		//return 0.;
	}
	default:
	{
		double error_sum = 0.0;
		const int lsize = model->shape.back();
		//ShowVector::show(desiredOutput);
		//printf("\n");
		for (int j = 0; j < lsize; j++) {
			//printf("%f, %f\n", desiredOutput[j], output[j]);
			error_sum += (output[j] - desiredOutput[j]) * (output[j] - desiredOutput[j]);
		}
		return(error_sum / double(lsize));
	}
	}
}

// This function contains all the derivatives of the cost functions (now only the mean squared error
// function). It calculates how much the calculated output values should change according to the
// desired output values. It calculates that value for a single output node ('j').
double ModelFunctions::cost_derivative(Model* model, std::vector<double> output, std::vector<double> desiredOutput, const int j, const int i)
{
	switch (i)
	{
	case 1:
	{
		// TODO: Add more cost functions
		return (0.);
	}
	default:
		const int lsize = model->shape.back();
		//printf("\n%d: %f, %f, %f", j, desiredOutput[j], output[j], double(lsize));
		return((output[j] - desiredOutput[j]) * 2 / double(lsize));
	}
}

// dCdl contains how much the calculated value stored in a specific node should change according to
// the nodes in the layer after it. The function starts at the output layer, taking the cost
// derivative of each output node. This way it has a vector of how the calculated values of each
// node should change according to the desired output. Next will be calculated how the values of
// each node in the next layer should change according to the nodes in the layer before it. This
// continues until it has been calculated for all nodes. This is then returned.
std::vector<std::vector<double>> ModelFunctions::dCdl(Model* model, std::vector<std::vector<double>> nodevals, std::vector<double> desiredOutput, int curr_layer = 0, std::vector<std::vector<double>> ret = std::vector<std::vector<double>>())
{
	std::vector<double> s = std::vector<double>();

	// When the function is run for the first time, it should start by calculating how the
	// calculated values of the output nodes should change according to the desired output and store
	// it in a vector.
	const int lsize = model->shape.back();
	if (curr_layer == 0) {
		std::vector<double> output = nodevals.back();
		std::vector<double>(lsize).swap(s);
		for (int j = 0; j < lsize; j++)
			s[j] = cost_derivative(model, output, desiredOutput, j, -1);
		ret.insert(ret.begin(), s);
	}
	else {
		s = ret.front();
	}

	// Now the first layer is calculated, the next one is up. This is where it will be increased
	// every time.
	curr_layer++;

	// Some variables that will store data that will have to be used more than once. This will save
	// a calculation time.
	const size_t size_1 = model->shape.size() - 1;
	const int currlsize = model->shape[size_1 - curr_layer];
	const int prevlsize = model->shape[size_1 - curr_layer + 1];

	if (curr_layer == size_1)
		return ret;

	// Initialisation of the vector in which the sums of the current layer will be stored.
	std::vector<double> s2 = std::vector<double>(currlsize);

	// Calculate the z vector or the current layer.
	std::vector<double> z = flatten(matmul(model->weights[size_1 - curr_layer], nodevals[size_1 - curr_layer]));
	int c = 0;
	for (auto& v : z) {
		v += model->biases[size_1 - curr_layer][c++];
	}
	const size_t sizez = z.size();

	std::vector<double> dldz = activation_derivative(z, model->actFuncts[size_1 - curr_layer]);

	// Loop through each node of the current layer.
#pragma omp parallel for
	for (int j = 0; j < currlsize; j++) {
		// 'sum' will be used to store the sum of how much the calculated value of each node in the
		// current layer should change according to each node in the layer before it.
		// Reset the sum to 0.
		double sum = 0.;

		// Loop through each node of the previous layer.
		for (int n = 0; n < prevlsize; n++) {
			// --------------------------------- Incorrect ---------------------------------
			// Should be model->weights[size_1 - curr_layer][n][j]
			// used to be nodevals[size_1 - curr_layer][n]
			double dzdl = model->weights[size_1 - curr_layer][n][j];
			// Calculate how node 'j' thinks node 'i' should change (dC/dl_j * dl/dz_j * dz_j/dl_i)
			// and add it to the sum.
			sum += s[n] * dldz[n] * dzdl;
		}

		// Expand the vector in which the current sums will be stored which will be used for
		// next time.
		s2[j] = sum;
	}

	ret.insert(ret.begin(), s2);
	return(dCdl(model, nodevals, desiredOutput, curr_layer, ret));
}

void ModelFunctions::backpropagation(Model* model, std::vector<std::vector<double>> nodevals, std::vector<double> desiredOutput, std::vector<std::vector<std::vector<double>>>& batch_dCdw_i, std::vector<std::vector<double>>& batch_dCdb_i)
{
	const size_t size_1 = model->shape.size() - 1;

	// Initialize vectors that will hold the gradient of the cost function w.r.t. the weights and biases with the shape they need.
	std::vector<std::vector<double>> dCdb(size_1);
	std::vector<std::vector<std::vector<double>>> dCdw(size_1);

	// A variable which will store data that will have to be used more than once. This will save a lot of calculation time.
	std::vector<std::vector<double>> dCdls = dCdl(model, nodevals, desiredOutput);

	// Loop through each layer starting from the front - near the input (layer 1). There is no reason for this other than convenience.
#pragma omp parallel for
	for (int l = 0; l < size_1; l++) {
		// Loop through the nodes of the current layer.
		const int lsize1 = model->shape[l + 1];
		const int lsize = model->shape[l];

		// Calculate the z vector or the current layer.
		std::vector<double> z = flatten(matmul(model->weights[l], nodevals[l]));
		int c = 0;
		for (auto& v : z) {
			v += model->biases[l][c++];
		}

		std::vector<double> dldz = activation_derivative(z, model->actFuncts[l]);

		std::vector<double> temp_dCdb(lsize1);
		std::vector<std::vector<double>> temp_dCdw(lsize1);
		for (int j = 0; j < lsize1; j++) {
			double dCdl_lj = dCdls[l][j];

			// With this, the gradient of the cost function w.r.t. the bias of the current node can be calculated by taking how much the calculated value stored the current node should change according to the nodes in the layer after it and multiplying it by the derivative of the activation function of z.
			temp_dCdb[j] = dCdl_lj * dldz[j];

			// Next, loop through each individual weight connected to the current node.
			std::vector<double> temp_temp_dCdw(lsize);
			for (int k = 0; k < lsize; k++) {
				// Then, the gradient of the cost function w.r.t. the current weight can be calculated done similarly to the way dCdb is calculated, but then with the 'z' for a specific weight and all multiplied by the node in the layer before it to which the weight is connected.
				double dzdw = nodevals[l][k];
				temp_temp_dCdw[k] = dCdl_lj * dldz[j] * dzdw;
			}
			temp_dCdw[j] = temp_temp_dCdw;
		}
		dCdb[l] = temp_dCdb;
		dCdw[l] = temp_dCdw;
	}

	// Now, the gradients of this training example will be stored with the others of its batch.
	batch_dCdw_i = dCdw;
	batch_dCdb_i = dCdb;
}

void ModelFunctions::optimize(Model* model, const double learning_rate, std::vector<std::vector<std::vector<std::vector<double>>>> batch_dCdw, std::vector<std::vector<std::vector<double>>> batch_dCdb)
{
	const size_t batchSize = batch_dCdb.size();
	// Check if the batches are of equal size.
	if (batchSize != batch_dCdw.size()) {
		printf("Invalid batches. Not equal in size or empty");
		return;
	}

	// Check if batchSize is valid. This prevents errors like division by zero later on.
	if (batchSize <= 0) {
		printf("ERROR AT optimize(): INVALID batchSize: %d", (int)batchSize);
		return;
	}

	const size_t lsize_1 = model->shape.size() - 1;

	// Initialize vectors that will hold the average gradient of the cost function w.r.t. the weights and biases with the shape they need.
	std::vector<std::vector<double>> avrg_dCdb(lsize_1);
	std::vector<std::vector<std::vector<double>>> avrg_dCdw(lsize_1);

	// Sum up each gradient of the training examples in this batch.
#pragma omp parallel for
	for (int l = 0; l < lsize_1; l++) {
		const int lsize = model->shape[l];
		const int lsize1 = model->shape[l + 1];
		std::vector<double> temp_avrg_dCdb(lsize1, 0);
		std::vector<std::vector<double>> temp_avrg_dCdw(lsize1);
		for (int j = 0; j < lsize1; j++) {
			temp_avrg_dCdb[j] = 0;
			for (int batch = 0; batch < batchSize; batch++) {
				temp_avrg_dCdb[j] += batch_dCdb[batch][l][j] / (double)batchSize;
			}
			std::vector<double> temp_temp_avrg_dCdw(lsize, 0);
			for (int k = 0; k < lsize; k++) {
				for (int batch = 0; batch < batchSize; batch++) {
					temp_temp_avrg_dCdw[k] += batch_dCdw[batch][l][j][k] / (double)batchSize;
				}
			}
			temp_avrg_dCdw[j] = temp_temp_avrg_dCdw;
		}
		avrg_dCdb[l] = temp_avrg_dCdb;
		avrg_dCdw[l] = temp_avrg_dCdw;
	}
	//printf("avrg_dCdw");
	//ShowVector::show(avrg_dCdw);
	//printf("\n");

	// Divide it now by the number of training examples in this batch. This will be the average.
	// Now, it is time for the weights and#p biases to be changed in proportion to the negative gradient of the cost function
#pragma omp parallel for
	for (int l = 0; l < lsize_1; l++) {
		const size_t size = avrg_dCdb[l].size();
		for (size_t j = 0; j < size; j++) {
			model->biases[l][j] -= learning_rate * avrg_dCdb[l][j];
		}
	}
#pragma omp parallel for
	for (int l = 0; l < lsize_1; l++) {
		const size_t size1 = avrg_dCdw[l].size();
		for (size_t j = 0; j < size1; j++) {
			const size_t size2 = avrg_dCdw[l][j].size();
			for (size_t k = 0; k < size2; k++) {
				model->weights[l][j][k] -= learning_rate * avrg_dCdw[l][j][k];
			}
		}
	}
}

void ModelFunctions::train(Model* model, const int iterations, TrainingData* TD, const double learning_rate = 1)
{
	std::vector<std::vector<std::vector<std::vector<double>>>> batched_training_data = TD->batched_training_data;
	//std::vector<double> costs;

	for (int it = 1; it <= iterations; it++) {
		std::vector<double> c;
		for (std::vector<std::vector<std::vector<double>>> batch : batched_training_data) {
			const size_t bsize = batch.size();
			std::vector<std::vector<std::vector<std::vector<double>>>> batch_dCdw(bsize);
			std::vector<std::vector<std::vector<double>>> batch_dCdb(bsize);
#pragma omp parallel for
			for (int i = 0; i < bsize; i++) {
				backpropagation(model, getNodevals(model, batch[i][0]), batch[i][1], batch_dCdw[i], batch_dCdb[i]);
			}
			optimize(model, learning_rate, batch_dCdw, batch_dCdb);
		}
		//costs.push_back(1.0 * std::accumulate(c.begin(), c.end(), .0) / c.size());
	}
	//ShowVector::show(costs);
	//printf("\n");
}

template<typename T>
void freeOutT(T* out)
{
	delete out;
}

double* feedforward(Model* model, double* input)
{
	std::vector<double> in;
	for (int i = 0; i < model->shape[0]; i++)
		in.push_back(input[i]);

	std::vector<double> vout = ModelFunctions::feedforward(model, in);

	double* out = new double[vout.size() + 1];
	int i = 0;
	for (auto& n : vout)
		out[i++] = n;
	return out;
}

void freeOutD(double* out)
{
	freeOutT(out);
}

void freeOutI(unsigned int* out)
{
	freeOutT(out);
}

void freeOutM(Model* out)
{
	freeOutT(out);
}

void trainFromFile(Model* model, const unsigned int iterations, const double learning_rate, const char* path_to_file, const unsigned int batch_size, const double percentage_of_data_not_to_train, const int training_data_seed)
{
	TrainingData* ShapedTD = new TrainingData(path_to_file, batch_size, percentage_of_data_not_to_train, training_data_seed);
	ModelFunctions::train(model, iterations, ShapedTD, learning_rate);
}

void trainFromArray(Model* model, const unsigned int iterations, const double learning_rate, const double* trainingDataArray, const unsigned int trainingDataSamples, const unsigned int batch_size, const double percentage_of_data_not_to_train, const int training_data_seed)
{
	std::vector<std::vector<std::vector<double>>> traingingData;
	int i = 0;
	for (unsigned int b = 0; b < trainingDataSamples; b++) {
		std::vector<std::vector<double>> temp;
		std::vector<double> input;
		for (int inp = 0; inp < model->shape[0]; inp++) {
			input.push_back(trainingDataArray[i++]);
		}
		temp.push_back(input);
		std::vector<double> output;
		for (int out = 0; out < model->shape[model->shape.size() - 1]; out++) {
			output.push_back(trainingDataArray[i++]);
		}
		temp.push_back(output);
		traingingData.push_back(temp);
	}

	TrainingData* ShapedTD = new TrainingData(traingingData, batch_size, percentage_of_data_not_to_train, training_data_seed);
	ModelFunctions::train(model, iterations, ShapedTD, learning_rate);
}