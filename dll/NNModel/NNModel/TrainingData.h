#pragma once

#include <random>
#include <vector>
#include <string>

class TrainingData
{
public:
	void replaceAll(std::string& str, const std::string& from, const std::string& to);
	std::vector<std::vector<std::vector<double>>> training_data_parser(std::string filename);
	std::vector<std::vector<std::vector<std::vector<double>>>> split_training_data(const std::vector<std::vector<std::vector<double>>>& arr, const size_t k);
	TrainingData(std::vector<std::vector<std::vector<double>>> training_data, const int batch_size, const double percentage_of_data_not_to_train);
	TrainingData(std::vector<std::vector<std::vector<double>>> training_data, const int batch_size, const double percentage_of_data_not_to_train, int seed);
	TrainingData(std::string filename, const int batch_size, const double percentage_of_data_not_to_train);
	TrainingData(std::string filename, const int batch_size, const double percentage_of_data_not_to_train, int seed);

	std::vector<std::vector<std::vector<std::vector<double>>>> batched_training_data;
	std::vector<std::vector<std::vector<double>>> untrainded_data;

private:
	std::mt19937 gen;
};
