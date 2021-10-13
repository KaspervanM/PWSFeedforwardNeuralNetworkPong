#include "pch.h"
#include "ModelStorage.h"
#include <fstream>
#include <sstream>

// Save a model to a file.
bool ModelStorage::save(Model* model, std::string filename)
{
	std::ofstream file;
	file.open(filename);
	if (file.is_open()) {
		file << "{";
		for (size_t l = 0; l < model->weights.size(); l++) {
			file << "{";
			for (size_t n = 0; n < model->weights[l].size(); n++) {
				file << "{";
				for (size_t w = 0; w < model->weights[l][n].size(); w++) {
					if (w < model->weights[l][n].size() - 1) file << model->weights[l][n][w] << ",";
					else file << model->weights[l][n][w];
				}
				if (n < model->weights[l].size() - 1) file << "},";
				else file << "}";
			}
			if (l < model->weights.size() - 1) file << "},";
			else file << "}";
		}
		file << "}";

		file << std::endl;

		file << "{";
		for (size_t l = 0; l < model->biases.size(); l++) {
			file << "{";
			for (size_t n = 0; n < model->biases[l].size(); n++) {
				if (n < model->biases[l].size() - 1) file << model->biases[l][n] << ",";
				else file << model->biases[l][n];
			}
			if (l < model->biases.size() - 1) file << "},";
			else file << "}";
		}
		file << "}";

		file << std::endl;
		file << "{";
		for (size_t l = 0; l < model->actFuncts.size(); l++) {
			if (l < model->actFuncts.size() - 1) file << (int)model->actFuncts[l] << ",";
			else file << (int)model->actFuncts[l];
		}
		file << "}";

		file.close();

		return true;
	}
	return false;
}

bool ModelStorage::load(Model* model, std::string filename)
{
	std::ifstream file;
	file.open(filename);

	if (file.is_open()) {
		std::vector<std::vector<std::vector<double>>>().swap(model->weights);
		std::vector<std::vector<double>>().swap(model->biases);
		std::vector<act>().swap(model->actFuncts);
		std::vector<int>().swap(model->shape);

		std::string line;
		std::getline(file, line);
		replaceAll(line, " ", "");
		//replaceAll(line, "\n", "");
		//replaceAll(line, "\r", "");
		replaceAll(line, "\t", "");
		replaceAll(line, "]", "}");
		replaceAll(line, "[", "{");
		replaceAll(line, ")", "}");
		replaceAll(line, "(", "{");
		replaceAll(line, "}},", ":");
		replaceAll(line, "},", ";");
		replaceAll(line, "{", "");
		replaceAll(line, "}", "");
		replaceAll(line, ",;", ";");
		replaceAll(line, ",:", ":");
		replaceAll(line, ";,", ";");
		replaceAll(line, ":,", ":");
		//std::cout<<line;
		{
			std::stringstream ss(line);
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
				model->weights.push_back(temp);
			}
		}

		line = "";
		std::getline(file, line);
		replaceAll(line, " ", "");
		//replaceAll(line, "\n", "");
		//replaceAll(line, "\r", "");
		replaceAll(line, "\t", "");
		replaceAll(line, "]", "}");
		replaceAll(line, "[", "{");
		replaceAll(line, ")", "}");
		replaceAll(line, "(", "{");
		replaceAll(line, "}},", ":");
		replaceAll(line, "},", ";");
		replaceAll(line, "{", "");
		replaceAll(line, "}", "");
		replaceAll(line, ",;", ";");
		replaceAll(line, ",:", ":");
		replaceAll(line, ";,", ";");
		replaceAll(line, ":,", ":");
		{
			std::stringstream ss(line);
			std::vector<std::string> split1;

			while (ss.good())
			{
				std::string substr;
				std::getline(ss, substr, ';');
				split1.push_back(substr);
			}

			for (std::string part : split1) {
				std::vector<double> temp;
				std::stringstream temp_ss(part);
				while (temp_ss.good())
				{
					std::string substr;
					std::getline(temp_ss, substr, ',');
					temp.push_back(std::stof(substr));
				}
				model->biases.push_back(temp);
			}
		}

		line = "";
		std::getline(file, line);
		replaceAll(line, " ", "");
		//replaceAll(line, "\n", "");
		//replaceAll(line, "\r", "");
		replaceAll(line, "\t", "");
		replaceAll(line, "]", "}");
		replaceAll(line, "[", "{");
		replaceAll(line, ")", "}");
		replaceAll(line, "(", "{");
		replaceAll(line, "}},", ":");
		replaceAll(line, "},", ";");
		replaceAll(line, "{", "");
		replaceAll(line, "}", "");
		replaceAll(line, ",;", ";");
		replaceAll(line, ",:", ":");
		replaceAll(line, ";,", ";");
		replaceAll(line, ":,", ":");
		{
			std::stringstream ss(line);

			while (ss.good())
			{
				std::string substr;
				std::getline(ss, substr, ',');
				model->actFuncts.push_back((act)std::stoi(substr));
			}
		}
		file.close();

		model->shape.push_back(model->weights[0][0].size());
		for (std::vector<double> s : model->biases) {
			model->shape.push_back(s.size());
		}

		return true;
	}
	return false;
}

void ModelStorage::replaceAll(std::string& str, const std::string& from, const std::string& to)
{
	if (from.empty())
		return;
	size_t start_pos = 0;
	while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
	}
}

bool saveModel(Model* model, const char* filename)
{
	return ModelStorage::save(model, filename);
}

void loadModel(Model* model, const char* filename)
{
	ModelStorage::load(model, filename);
}