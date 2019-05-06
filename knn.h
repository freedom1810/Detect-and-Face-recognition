#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <unordered_map>
#include <iostream>
#include <algorithm>

using namespace dlib;
using namespace std;

bool comp(pair<int, int> a, pair<int, int> b);

unsigned long predict(
	matrix<float, 0, 1> target,
	std::vector<matrix<float, 0, 1>> samples,
	std::vector<unsigned long> labels
);