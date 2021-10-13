#include "pch.h"
#include "TrainingData.h"
#include "ShowVector.h"

#include <cassert>
#include <fstream>
#include <sstream>
#include <stdexcept>

void TrainingData::replaceAll(std::string& str, const std::string& from, const std::string& to)
{
	if (from.empty())
		return;
	size_t start_pos = 0;
	while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
	}
}

std::vector<std::vector<std::vector<double>>> TrainingData::training_data_parser(std::string filename)
{
	std::ifstream file;
	file.open(filename, std::ios::binary);

	if (file.is_open()) {
		std::vector<std::vector<std::vector<double>>> training_data;
		std::string content;
		content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
		replaceAll(content, " ", "");
		replaceAll(content, "\n", "");
		replaceAll(content, "\r", "");
		replaceAll(content, "\t", "");
		replaceAll(content, "]", "}");
		replaceAll(content, "[", "{");
		replaceAll(content, ")", "}");
		replaceAll(content, "(", "{");
		replaceAll(content, "}},", ":");
		replaceAll(content, "},", ";");
		replaceAll(content, "{", "");
		replaceAll(content, "}", "");
		replaceAll(content, ",;", ";");
		replaceAll(content, ",:", ":");
		replaceAll(content, ";,", ";");
		replaceAll(content, ":,", ":");
		while (!std::isdigit(content.back())) content.pop_back();
		std::stringstream ss(content);
		std::vector<std::string> split1;

		while (ss.good())
		{
			std::string substr;
			std::getline(ss, substr, ':');
			split1.push_back(substr);
		}

		std::vector<std::vector<std::string>> split2;
		for (std::string part : split1) {
			std::vector<std::string> temp;
			std::stringstream temp_ss(part);
			while (temp_ss.good())
			{
				std::string substr;
				std::getline(temp_ss, substr, ';');
				temp.push_back(substr);
			}
			split2.push_back(temp);
		}

		for (std::vector<std::string> part : split2) {
			std::vector<std::vector<double>> temp;
			for (std::string part2 : part) {
				std::vector<double> temp2;
				std::stringstream temp_ss(part2);
				while (temp_ss.good())
				{
					std::string substr;
					std::getline(temp_ss, substr, ',');
					temp2.push_back(std::stof(substr));
				}
				temp.push_back(temp2);
			}
			training_data.push_back(temp);
		}
		return training_data;
	}
	printf("\nFile not found\n");
}

std::vector<std::vector<std::vector<std::vector<double>>>> TrainingData::split_training_data(const std::vector<std::vector<std::vector<double>>>& arr, const size_t k)
{
	if (k >= arr.size()) return std::vector<std::vector<std::vector<std::vector<double>>>>{arr};
	std::vector<std::vector<std::vector<std::vector<double>>>> result;
	assert(k > 0);
	result.reserve(arr.size() / k);

	for (size_t i = 0; i < arr.size(); i += k) {
		size_t begin = i;
		size_t end = min(arr.size(), i + k);
		result.emplace_back(arr.begin() + begin, arr.begin() + end);
	}
	return result;
}

TrainingData::TrainingData(std::string filename, const int batch_size, const double percentage_of_data_not_to_train)
	: TrainingData(filename, batch_size, percentage_of_data_not_to_train, -1)
{}

TrainingData::TrainingData(std::string filename, const int batch_size, const double percentage_of_data_not_to_train, const int seed)
	: TrainingData(training_data_parser(filename), batch_size, percentage_of_data_not_to_train, seed)
{}

TrainingData::TrainingData(std::vector<std::vector<std::vector<double>>> training_data, const int batch_size, const double percentage_of_data_not_to_train)
	: TrainingData(training_data, batch_size, percentage_of_data_not_to_train, -1)
{}

TrainingData::TrainingData(std::vector<std::vector<std::vector<double>>> training_data, const int batch_size, const double percentage_of_data_not_to_train, const int seed)
{
	if (seed == -1) gen = std::mt19937(std::random_device{}());
	else gen = std::mt19937(seed);

	if (seed != -2) {
		std::shuffle(std::begin(training_data), std::end(training_data), gen);
		std::shuffle(std::begin(training_data), std::end(training_data), gen);
	}

	untrainded_data.assign(training_data.begin(), training_data.begin() + (int)((double)training_data.size() * percentage_of_data_not_to_train / 100.0));
	std::vector<std::vector<std::vector<double>>> training_data_left(training_data.begin() + (int)((double)training_data.size() * percentage_of_data_not_to_train / 100.0), training_data.end());

	batched_training_data = split_training_data(training_data_left, batch_size);
}