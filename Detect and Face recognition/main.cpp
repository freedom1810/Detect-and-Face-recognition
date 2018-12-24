//#include <typeinfo>

// ----------------------------------------------------------------------------------------
// nhận ảnh tử webcam
#include <iostream>
#include <fstream>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing.h>

#include "detect_recoginze.h"
#include "load_image.h"
#include "svm.h"

using namespace dlib;
using namespace std;
using namespace cv;
// ----------------------------------------------------------------------------------------

void process()
try
{
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cerr << "Unable to connect to camera" << endl;
	}

	image_window win;

	// Grab and process frames until the main window is closed by the user.
	while (!win.is_closed())
	{
		// Grab a frame
		cv::Mat input;

		if (!cap.read(input))
		{
			break;
		}

		cv_image<bgr_pixel> image(input);

		if (win.is_closed())
		{
			matrix<rgb_pixel> img;
			//load ảnh ngoài
			//load_image(img, "input.jpg");

			//load ảnh từ camera
			assign_image(img, image);

			//load các mặt trong ảnh, convert ra 128
			std::vector<matrix<float, 0, 1>> face_descriptors = detect_camera(img);

			cout << "so khuan mat phat hien ra la: " << face_descriptors.size() << endl;
			unsigned long a;
			predict(face_descriptors);
			//a = predict(face_descriptors[0], data_face_descriptors, labels);
			//cout << a << endl;

			cout << "hit enter to terminate" << endl;
			cin.get();
		}

		win.clear_overlay();
		win.set_image(image);
	}
}
catch (std::exception& e)
{
	cout << e.what() << endl;
}

// -------------------
int main(int argc, char** argv)
{
	string ans;
	cout << "Do you want to train ? [Y/n]: ";
	cin >> ans;
	if (ans == "y")
	{
		// load anh 
		auto objs = load_objects_list(argv[1]);

		cout << "objs.size(): " << objs.size() << endl;

		//test dia chỉ của ảnh 
		//cout << objs[0][0]<<endl;

		std::vector<matrix<rgb_pixel>> images;
		std::vector<unsigned long> labels;
		dlib::rand rnd(time(0));
		size_t so_anh_moi_nguoi = 9;
		size_t so_nguoi_train = 7;
		load_mini_batch(so_nguoi_train, so_anh_moi_nguoi, rnd, objs, images, labels);
		//cout << labels[10] << endl;

		//--------------------------------------
		//show ảnh train của mỗi label, set label
		std::vector<image_window> set_labels(so_nguoi_train);

		for (int i = 0; i < so_nguoi_train; i++)
		{
			std::vector<matrix<rgb_pixel>> test(so_anh_moi_nguoi);
			std::copy(images.begin() + i * so_anh_moi_nguoi, images.begin() + (i + 1) * so_anh_moi_nguoi, test.begin());


			set_labels[i].set_title("label " + cast_to_string(i));
			set_labels[i].set_image(tile_images(test));
		}
		//--------------------------------------

		//lấy ảnh của mỗi người rồi -> thành 128D vector
		std::vector<matrix<float, 0, 1>> data_face_descriptors = detect(images);
		//cout << data_face_descriptors.size() << endl;

		train_data(data_face_descriptors, labels);
	}

	//nhận ảnh tử webcam và nhận dạng
	process();

	cout << "hit enter to terminate" << endl;
	cin.get();
}

// -------------------

std::vector<matrix<rgb_pixel>> jitter_image(
	const matrix<rgb_pixel>& img
)
{
	// All this function does is make 100 copies of img, all slightly jittered by being
	// zoomed, rotated, and translated a little bit differently. They are also randomly
	// mirrored left to right.
	thread_local dlib::rand rnd;

	std::vector<matrix<rgb_pixel>> crops;
	for (int i = 0; i < 100; ++i)
		crops.push_back(jitter_image(img, rnd));

	return crops;
}

// ----------------------------------------------------------------------------------------

