#include "pch.h"
#include "model.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

namespace array2bmp
{
#define MAXNODES 50
	struct Image {
		int w, h;				// Size of the scaled image.
		int width, height;		// Size of the bitmap.
		int padding;			// Padding in pixels required per horizontal pixel line.
		int ppn, ppl;			// Pixels per node and pisels per layer.
	};
	bool drawSquare(uint8_t* imageData, Image image, int* imageData_buff, int x, int y, int r, int g)
	{
		if (g < 0 || g > 256)
			return false;

		for (int _h = 0; _h <= 2 * r; _h++) {
			for (int _w = 0; _w <= 2 * r; _w++) {
				int _x = x - (r - _w);
				int _y = y - (r - _h);
				int i = _x + image.width * _y;
				if (i >= 0 && i < image.width * image.height) {
					imageData_buff[i] = g;
				}
			}
		}
		return true;
	}

	bool drawLine(uint8_t* imageData, Image image, int* imageData_buff, int x1, int y1, int x2, int y2, int g)
	{
		double a = (double(y2) - double(y1)) / (double(x2) - double(x1));
		double b = y1 - a * x1;
		int prev = 0;
		for (int x = min(x1, x2); x <= max(x1, x2); x++) {
			int y = int(a * x + b);
			int fillc = prev - y;
			if (prev != 0 && fillc != 0) {
				for (int p = min(0, fillc); p < max(0, fillc); p++) {
					int i = x + image.width * y + image.width * p;
					if (i >= 0 && i < image.width * image.height) {
						imageData_buff[i] += g;
					}
				}
			}
			else {
				int i = x + image.width * y;
				if (i >= 0 && i < image.width * image.height) {
					imageData_buff[i] += g;
				}
			}
			prev = y;
		}

		return true;
	}

	bool generateImageData(Model* model, Image image, uint8_t* imageData)
	{
		if (model->weights.size() == 0)
			return false;

		int* imageData_buff = new int[image.width * image.height];
		for (int i = 0; i < image.width * image.height; i++) {
			imageData_buff[i] = 0;
		}

		// Prevent too large layers from unbalancing the image.
		int* layers = new int[model->shape.size()];
		for (int l = 0; l < model->shape.size(); l++) {
			layers[l] = min(model->shape[l], MAXNODES);
		}

		int margin = int(double(image.ppn) * 1.5 + .5) - 1;

		int heightinppn = image.height / image.ppn;
		for (int l = 0; l < model->shape.size() - 1; l++) {
			int deltalayerinppn = 2 * (layers[l + 1] - 1);
			int offsetinppn = (heightinppn - 3 - deltalayerinppn) / 2;
			int y1_bottom = -int(double(image.ppn) / 2. + .5) + image.ppn * offsetinppn;
			deltalayerinppn = 2 * (layers[l] - 1);
			offsetinppn = (heightinppn - 3 - deltalayerinppn) / 2;
			int y2_bottom = -int(double(image.ppn) / 2. + .5) + image.ppn * offsetinppn;
			for (int n = 1; n <= layers[l + 1]; n++) {
				int x1 = margin - int(double(image.ppn) / 2. + .5) + (l + 1) * image.ppl;
				int y1 = y1_bottom + n * image.ppn * 2;
				for (int w = 1; w <= layers[l]; w++) {
					int x2 = margin + int(double(image.ppn) / 2.) + l * image.ppl;
					int y2 = y2_bottom + w * image.ppn * 2;
					drawLine(imageData, image, imageData_buff, x1, y1, x2, y2, int(model->weights[l][n - 1][w - 1] * 1000 + .5));
				}
			}
		}

		// Now there could be negative values in the array lower than -127, this is not allowed. So, all the values are scaled so that the lowest value is -127.
		// The same goes for values above 127, they are scaled so that the highest value is 127.
		int min = 0;
		int max = 0;
		for (int i = 0; i < image.width * image.height; i++) {
			if (imageData_buff[i] < min)
				min = imageData_buff[i];
			if (imageData_buff[i] > max)
				max = imageData_buff[i];
		}
		const int scale = max(-min, max);
		for (int i = 0; i < image.width * image.height; i++) {
			imageData_buff[i] *= 127;
			imageData_buff[i] /= (scale != 0 ? scale : 1);
		}

		// Now the array is filled with values relative to 0, this should change to 128 so it can be put in a bitmap.
		for (int i = 0; i < image.width * image.height; i++) {
			imageData_buff[i] += 128;
		}

		// Draw the nodes.
		for (int l = 0; l < model->shape.size(); l++) {
			int deltalayerinppn = 2 * (layers[l] - 1);
			int offsetinppn = (heightinppn - 3 - deltalayerinppn) / 2;
			int y_bottom = -int(double(image.ppn) / 2. + .5) + image.ppn * offsetinppn;
			for (int n = 1; n <= layers[l]; n++) {
				drawSquare(imageData, image, imageData_buff, margin + l * image.ppl, y_bottom + n * image.ppn * 2, int(double(image.ppn) / 2.), 255);
			}
		}

		// Include image.padding.
		for (int _h = 0; _h < image.height; _h++) {
			for (int _w = 0; _w < image.width; _w++) {
				imageData[_w + (image.width + image.padding) * _h] = imageData_buff[_w + (image.width) * _h];
			}
			for (int _p = 0; _p < image.padding; _p++) {
				imageData[image.width + _p + (image.width + image.padding) * _h] = 48;
			}
		}

		delete[] imageData_buff;
		delete[] layers;

		return true;
	}

	//--------------------------------------------------------------------------
	// This little helper is to write little-endian values to file.
	//
	struct lwrite
	{
		unsigned long value;
		unsigned      size;
		lwrite(unsigned long value, unsigned size) :
			value(value), size(size)
		{ }
	};

	//--------------------------------------------------------------------------
	inline std::ostream& operator << (std::ostream& outs, const lwrite& v)
	{
		unsigned long value = v.value;
		for (unsigned cntr = 0; cntr < v.size; cntr++, value >>= 8)
			outs.put(static_cast <char> (value & 0xFF));
		return outs;
	}

	//--------------------------------------------------------------------------
	// Take an integer array and convert it into a color image.
	//
	// This first version takes an array of array style of array:
	//   int* a[ 10 ]
	//
	// The second, overloaded version takes a flat C-style array:
	//   int a[ 10 ][ 10 ]
	//
	template <typename IntType>
	bool intarray2bmp(
		const std::string& filename,
		IntType** intarray,
		unsigned           rows,
		unsigned           columns,
		IntType            min_value,
		IntType            max_value
	) {
		// This is the difference between each color based upon
		// the number of distinct values in the input array.
		double granularity = 360.0 / ((double)(max_value - min_value) + 1);

		// Open the output BMP file
		std::ofstream f(filename.c_str(),
			std::ios::out | std::ios::trunc | std::ios::binary);
		if (!f) return false;

		// Some basic
		unsigned long headers_size = 14  // sizeof( BITMAPFILEHEADER )
			+ 40; // sizeof( BITMAPINFOHEADER )
		unsigned long padding_size = (4 - ((columns * 3) % 4)) % 4;
		unsigned long pixel_data_size = rows * ((columns * 3) + padding_size);

		// Write the BITMAPFILEHEADER
		f.put('B').put('M');                           // bfType
		f << lwrite(headers_size + pixel_data_size, 4);  // bfSize
		f << lwrite(0, 2);  // bfReserved1
		f << lwrite(0, 2);  // bfReserved2
		f << lwrite(headers_size, 4);  // bfOffBits

		// Write the BITMAPINFOHEADER
		f << lwrite(40, 4);  // biSize
		f << lwrite(columns, 4);  // biWidth
		f << lwrite(rows, 4);  // biHeight
		f << lwrite(1, 2);  // biPlanes
		f << lwrite(24, 2);  // biBitCount
		f << lwrite(0, 4);  // biCompression=BI_RGB
		f << lwrite(pixel_data_size, 4);  // biSizeImage
		f << lwrite(0, 4);  // biXPelsPerMeter
		f << lwrite(0, 4);  // biYPelsPerMeter
		f << lwrite(0, 4);  // biClrUsed
		f << lwrite(0, 4);  // biClrImportant

		// Write the pixel data
		for (unsigned row = rows; row; row--)           // bottom-to-top
		{
			for (unsigned col = 0; col < columns; col++)  // left-to-right
			{
				unsigned char red, green, blue;
				//
				// This is how we convert an integer value to a color:
				// by mapping it evenly along the CIECAM02 hue color domain.
				//
				// http://en.wikipedia.org/wiki/Hue
				// http://en.wikipedia.org/wiki/hsl_and_hsv#conversion_from_hsv_to_rgb
				//
				// The following algorithm takes a few shortcuts since
				// both 'value' and 'saturation' are always 1.0.
				//
				red = green = blue = intarray[row - 1][col];
				/*double hue = (intarray[row - 1][col] - min_value) * granularity;
				int    H = (int)(hue / 60) % 6;
				double F = (hue / 60) - H;
				double Q = 1.0 - F;

#define c( x ) (255 * x)
				switch (H)
				{
				case 0:  red = c(1);  green = c(F);  blue = c(0);  break;
				case 1:  red = c(Q);  green = c(1);  blue = c(0);  break;
				case 2:  red = c(0);  green = c(1);  blue = c(F);  break;
				case 3:  red = c(0);  green = c(Q);  blue = c(1);  break;
				case 4:  red = c(F);  green = c(0);  blue = c(1);  break;
				default: red = c(1);  green = c(0);  blue = c(Q);
				}
#undef c*/

				f.put(static_cast <char> (blue))
					.put(static_cast <char> (green))
					.put(static_cast <char> (red));
			}

			if (padding_size) f << lwrite(0, padding_size);
		}

		// All done!
		return f.good();
	}

	//--------------------------------------------------------------------------
	template <typename IntType>
	bool intarray2bmp(
		const std::string& filename,
		IntType* intarray,
		unsigned           rows,
		unsigned           columns,
		IntType            min_value,
		IntType            max_value
	) {
		IntType** ia = new(std::nothrow) IntType * [rows];
		for (unsigned row = 0; row < rows; row++)
		{
			ia[row] = intarray + (row * columns);
		}
		bool result = intarray2bmp(
			filename, ia, rows, columns, min_value, max_value
		);
		delete[] ia;
		return result;
	}

	bool createBitmap(Model* model, const char* filename)
	{
		int maxnodesinlayer = *max_element(std::begin(model->shape), std::end(model->shape));
		maxnodesinlayer = min(maxnodesinlayer, MAXNODES);

		Image image;
		image.w = -1;
		image.h = -1;
		image.ppn = 3;
		image.ppn += (image.ppn % 2 + 1) % 2;
		image.ppl = maxnodesinlayer * image.ppn;
		image.width = 3 * image.ppn + (model->shape.size() - 1) * image.ppl;
		image.height = image.ppn + maxnodesinlayer * image.ppn * 2;

		// Include image.padding.
		image.padding = 4 - image.width % 4;
		if (image.padding % 4 == 0)
			image.padding = 0;

		uint8_t* imageData = new uint8_t[(image.width + image.padding) * image.height];
		generateImageData(model, image, imageData);

		uint8_t** imageData2 = new uint8_t * [image.height];
		for (int i = 0; i < image.height; i++) {
			imageData2[i] = new uint8_t[(image.width + image.padding)];
			for (int j = 0; j < (image.width + image.padding); j++)
				imageData2[i][j] = imageData[i * (image.width + image.padding) + j];
		}
		bool out = intarray2bmp(filename, imageData2, (uint8_t)image.height, (uint8_t)(image.width + image.padding), (uint8_t)0, (uint8_t)255);
		delete[] imageData;
		delete[] imageData2;

		return out;
	}
} // namespace intarray2bmp

extern "C" NNMODEL_API bool createBitmap(Model * model, const char* filename)
{
	return array2bmp::createBitmap(model, filename);
}

/*
CreateBitmap::~CreateBitmap()
{
	delete[] imageData;
}

void CreateBitmap::changeModel(Model* m)
{
	model = m;
	int maxnodesinlayer = *max_element(std::begin(model->shape), std::end(model->shape));
	maxnodesinlayer = min(maxnodesinlayer, MAXNODES);

	image.w = -1;
	image.h = -1;
	image.ppn = 3;
	image.ppn += (image.ppn % 2 + 1) % 2;
	image.ppl = maxnodesinlayer * image.ppn;
	image.width = 3 * image.ppn + (model->shape.size() - 1) * image.ppl;
	image.height = image.ppn + maxnodesinlayer * image.ppn * 2;

	// Include image.padding.
	image.padding = 4 - image.width % 4;
	if (image.padding % 4 == 0)
		image.padding = 0;
	imageData = new uint8_t[(image.width + image.padding) * image.height];
}

bool CreateBitmap::drawSquare(int* imageData_buff, int x, int y, int r, int g)
{
	if (g < 0 || g > 256)
		return false;

	for (int _h = 0; _h <= 2 * r; _h++) {
		for (int _w = 0; _w <= 2 * r; _w++) {
			int _x = x - (r - _w);
			int _y = y - (r - _h);
			int i = _x + image.width * _y;
			if (i >= 0 && i < image.width * image.height) {
				imageData_buff[i] = g;
			}
		}
	}
	return true;
}

bool CreateBitmap::drawLine(int* imageData_buff, int x1, int y1, int x2, int y2, int g)
{
	double a = (double(y2) - double(y1)) / (double(x2) - double(x1));
	double b = y1 - a * x1;
	int prev = 0;
	for (int x = min(x1, x2); x <= max(x1, x2); x++) {
		int y = int(a * x + b);
		int fillc = prev - y;
		if (prev != 0 && fillc != 0) {
			for (int p = min(0, fillc); p < max(0, fillc); p++) {
				int i = x + image.width * y + image.width * p;
				if (i >= 0 && i < image.width * image.height) {
					imageData_buff[i] += g;
				}
			}
		}
		else {
			int i = x + image.width * y;
			if (i >= 0 && i < image.width * image.height) {
				imageData_buff[i] += g;
			}
		}
		prev = y;
	}

	return true;
}

bool CreateBitmap::generateImageData()
{
	if (model->weights.size() == 0)
		return false;

	int* imageData_buff = new int[image.width * image.height];
	for (int i = 0; i < image.width * image.height; i++) {
		imageData_buff[i] = 0;
	}

	// Prevent too large layers from unbalancing the image.
	int* layers = new int[model->shape.size()];
	for (int l = 0; l < model->shape.size(); l++) {
		layers[l] = min(model->shape[l], MAXNODES);
	}

	int margin = int(double(image.ppn) * 1.5 + .5) - 1;

	int heightinppn = image.height / image.ppn;
	for (int l = 0; l < model->shape.size() - 1; l++) {
		int deltalayerinppn = 2 * (layers[l + 1] - 1);
		int offsetinppn = (heightinppn - 3 - deltalayerinppn) / 2;
		int y1_bottom = -int(double(image.ppn) / 2. + .5) + image.ppn * offsetinppn;
		deltalayerinppn = 2 * (layers[l] - 1);
		offsetinppn = (heightinppn - 3 - deltalayerinppn) / 2;
		int y2_bottom = -int(double(image.ppn) / 2. + .5) + image.ppn * offsetinppn;
		for (int n = 1; n <= layers[l + 1]; n++) {
			int x1 = margin - int(double(image.ppn) / 2. + .5) + (l + 1) * image.ppl;
			int y1 = y1_bottom + n * image.ppn * 2;
			for (int w = 1; w <= layers[l]; w++) {
				int x2 = margin + int(double(image.ppn) / 2.) + l * image.ppl;
				int y2 = y2_bottom + w * image.ppn * 2;
				drawLine(imageData_buff, x1, y1, x2, y2, int(model->weights[l + 1][n - 1][w - 1] * 1000 + .5));
			}
		}
	}

	// Now there could be negative values in the array lower than -127, this is not allowed. So, all the values are scaled so that the lowest value is -127.
	// The same goes for values above 127, they are scaled so that the highest value is 127.
	int min = 0;
	int max = 0;
	for (int i = 0; i < image.width * image.height; i++) {
		if (imageData_buff[i] < min)
			min = imageData_buff[i];
		if (imageData_buff[i] > max)
			max = imageData_buff[i];
	}
	const int scale = max(-min, max);
	for (int i = 0; i < image.width * image.height; i++) {
		imageData_buff[i] *= 127;
		imageData_buff[i] /= (scale != 0 ? scale : 1);
	}

	// Now the array is filled with values relative to 0, this should change to 128 so it can be put in a bitmap.
	for (int i = 0; i < image.width * image.height; i++) {
		imageData_buff[i] += 128;
	}

	// Draw the nodes.
	for (int l = 0; l < model->shape.size(); l++) {
		int deltalayerinppn = 2 * (layers[l] - 1);
		int offsetinppn = (heightinppn - 3 - deltalayerinppn) / 2;
		int y_bottom = -int(double(image.ppn) / 2. + .5) + image.ppn * offsetinppn;
		for (int n = 1; n <= layers[l]; n++) {
			drawSquare(imageData_buff, margin + l * image.ppl, y_bottom + n * image.ppn * 2, int(double(image.ppn) / 2.), 255);
		}
	}

	// Include image.padding.
	for (int _h = 0; _h < image.height; _h++) {
		for (int _w = 0; _w < image.width; _w++) {
			imageData[_w + (image.width + image.padding) * _h] = imageData_buff[_w + (image.width) * _h];
		}
		for (int _p = 0; _p < image.padding; _p++) {
			imageData[image.width + _p + (image.width + image.padding) * _h] = 48;
		}
	}

	delete[] imageData_buff;
	delete[] layers;

	return true;
}*/