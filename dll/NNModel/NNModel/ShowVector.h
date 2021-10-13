#pragma once
#include <iostream>
#include <vector>

class ShowVector
{
public:
	template<typename T>
	static void show(T vec)
	{
		std::cout << (double)vec;
	}

	// A recursive function able to print a vector of an arbitrary amount of dimensions
	template<typename T>
	static void show(std::vector<T> vec)
	{
		int size = vec.size();
		if (size <= 0) {
			std::cout << "invalid vector";
			return;
		}
		std::cout << '{';
		for (int l = 0; l < size - 1; l++) {
			show(vec[l]);
			std::cout << ',';
		}
		show(vec[size - 1]);
		std::cout << '}';
	}
};
