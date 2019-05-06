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

/*
Faces are connected in the graph if they are close enough.  Here we check if
the distance between two face descriptors is less than 0.6, which is the
decision threshold the network was trained to use.  Although you can
certainly use any other threshold you find useful.
	if (length(face_descriptors[i] - face_descriptors[j]) < 0.6)
		edges.push_back(sample_pair(i, j));
*/

bool comp(pair<int, int> a, pair<int, int> b) {
	return a.second < b.second;
}

unsigned long predict(
	matrix<float, 0, 1> target,
	std::vector<matrix<float, 0, 1>> samples,
	std::vector<unsigned long> labels
)
{
	//limit for classification
	//int k = 3;

	/*
	maximum value of Euclidean distance between samples
	Here we check if the distance between two face descriptors is less than 0.6, which is the
	decision threshold the network was trained to use.Although you can
	certainly use any other threshold you find useful.
	*/
	float thresh = 0.6;

	std::unordered_map<unsigned long, int> distances;

	//we load each person 5 vector image, it store in samples and labels
	for (int i = 0; i < samples.size(); i++)
	{
		if (length(target - samples[i]) < thresh)
		{
			if (distances.find(labels[i]) == distances.end())
			{
				distances[labels[i]] = 1;
			}
			else
			{
				distances[labels[i]] += 1;
			}
		}
	}

	int max_value = -1;
	unsigned long ans_label = -1;
	for (auto& elem : distances)
	{
		if (max_value < elem.second)
		{
			max_value = elem.second;
			ans_label = elem.first;
		}
	}

	return ans_label;
}