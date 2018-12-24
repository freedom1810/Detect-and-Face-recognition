// Our data will be 3-dimensional data

#include <typeinfo>

#include <dlib/svm_threaded.h>

#include <iostream>
#include <vector>

#include <dlib/rand.h>

using namespace std;
using namespace dlib;

typedef matrix<double, 128, 1> sample_type;

int train_data(
	std::vector<matrix<float, 0, 1>> data_face_descriptors,
	std::vector<unsigned long> labels
);

void generate_data(
	std::vector<sample_type>& samples,
	std::vector<matrix<float, 0, 1>> data_face_descriptors
);

void predict(
	std::vector<matrix<float, 0, 1>> face_descriptors
);