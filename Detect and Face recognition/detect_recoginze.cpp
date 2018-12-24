#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
	const matrix<rgb_pixel>& img
);

std::vector<matrix<float, 0, 1>> detect(
	std::vector<matrix<rgb_pixel>>& images
)
{
	//load model: find faces in the image we will need a face detector:
	frontal_face_detector detector = get_frontal_face_detector();
	// We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
	shape_predictor sp;
	deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
	// load the DNN responsible for face recognition.
	anet_type net;
	deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

	std::vector<matrix<rgb_pixel>> faces;
	for (int i = 0; i < images.size(); i++)
	{
		//cout << labels[i] << endl;
		//--------------------------------------------
		//convert du lieu anh moi label de kiem tra

		for (auto face : detector(images[i]))
		{
			auto shape = sp(images[i], face);
			matrix<rgb_pixel> face_chip;
			extract_image_chip(images[i], get_face_chip_details(shape, 150, 0.25), face_chip);
			faces.push_back(move(face_chip));
		}
		//--------------------------------------------
	}
	std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);

	return face_descriptors;
}

std::vector<matrix<float, 0, 1>> detect_camera(
	matrix<rgb_pixel>& img
)
{
	//load model: find faces in the image we will need a face detector:
	frontal_face_detector detector = get_frontal_face_detector();
	// We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
	shape_predictor sp;
	deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
	// load the DNN responsible for face recognition.
	anet_type net;
	deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

	std::vector<matrix<rgb_pixel>> faces;

	// Display the raw image on the screen
	image_window win(img);

	for (auto face : detector(img))
	{
		auto shape = sp(img, face);
		matrix<rgb_pixel> face_chip;
		extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
		faces.push_back(move(face_chip));
		// Also put some boxes on the faces so we can see that the detector is finding them
		win.add_overlay(face);
	}

	if (faces.size() == 0)
	{
		cout << "No faces found in image!" << endl;
	}

	std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);

	return face_descriptors;
}

