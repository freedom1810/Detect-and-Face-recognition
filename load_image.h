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
);

void load_mini_batch(
	const size_t num_people,     // how many different people to include
	const size_t samples_per_id, // how many images per person to select.
	dlib::rand& rnd,
	const std::vector<std::vector<string>>& objs,
	std::vector<matrix<rgb_pixel>>& images,
	std::vector<unsigned long>& labels
);
//------------------------------------------------------------------------
