//------------------------------------------------------------------------
// load folder image
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <algorithm> 

#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/misc_api.h>

using namespace dlib;
using namespace std;

std::vector<std::vector<string>> load_objects_list(
	const string& dir
)
{
	std::vector<std::vector<string>> objects;
	for (auto subdir : directory(dir).get_dirs())
	{
		std::vector<string> imgs;
		for (auto img : subdir.get_files())
			imgs.push_back(img);

		if (imgs.size() != 0)
			objects.push_back(imgs);
	}
	return objects;
}

void load_mini_batch(
	const size_t num_people,     // how many different people to include
	const size_t samples_per_id, // how many images per person to select.
	dlib::rand& rnd,
	const std::vector<std::vector<string>>& objs,
	std::vector<matrix<rgb_pixel>>& images,
	std::vector<unsigned long>& labels
)
{
	images.clear();
	labels.clear();
	DLIB_CASSERT(num_people <= objs.size(), "The dataset doesn't have that many people in it.");

	matrix<rgb_pixel> image;
	for (size_t i = 0; i < num_people; ++i)
	{
		size_t id = rnd.get_random_32bit_number() % objs.size();

		// lấy ảnh của người thứ i, random samples_per_id ảnh
		for (size_t j = 0; j < samples_per_id; ++j)
		{
			//const auto& obj = objs[id][rnd.get_random_32bit_number() % objs[id].size()];
			const auto& obj = objs[i][rnd.get_random_32bit_number() % objs[i].size()];
			load_image(image, obj);
			images.push_back(std::move(image));
			labels.push_back(i);
		}
	}

	// You might want to do some data augmentation at this point.  Here we do some simple
	// color augmentation.
	for (auto&& crop : images)
	{
		disturb_colors(crop, rnd);
		// Jitter most crops
		if (rnd.get_random_double() > 0.1)
			crop = jitter_image(crop, rnd);
	}

	// All the images going into a mini-batch have to be the same size.  And really, all
	// the images in your entire training dataset should be the same size for what we are
	// doing to make the most sense.  
	DLIB_CASSERT(images.size() > 0);
	for (auto&& img : images)
	{
		DLIB_CASSERT(img.nr() == images[0].nr() && img.nc() == images[0].nc(),
			"All the images in a single mini-batch must be the same size.");
	}
}
//------------------------------------------------------------------------#pragma once
