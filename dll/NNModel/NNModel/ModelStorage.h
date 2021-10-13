#pragma once
#include <string>
#include "Model.h"

class ModelStorage
{
public:
	// Save a model to a file.
	static bool save(Model* model, std::string filename);
	static bool load(Model* model, std::string filename);
	static void replaceAll(std::string& str, const std::string& from, const std::string& to);
};

extern "C" NNMODEL_API bool saveModel(Model * model, const char* filename);
extern "C" NNMODEL_API void loadModel(Model * model, const char* filename);