// Our data will be 3-dimensional data

#include <typeinfo>

#include <dlib/svm_threaded.h>

#include <iostream>
#include <vector>

#include <dlib/rand.h>

using namespace std;
using namespace dlib;

typedef matrix<double, 128, 1> sample_type;

void generate_data(
	std::vector<sample_type>& samples,
	std::vector<matrix<float, 0, 1>> data_face_descriptors
);

// ---------------------------

int train_data(
	std::vector<matrix<float, 0, 1>> data_face_descriptors,
	std::vector<unsigned long> labels
)
{
	/*
		std::vector<sample_type> samples;
		std::vector<string> labels;
	*/
	std::vector<sample_type> samples;
	// First, get our labeled set of training data
	generate_data(samples, data_face_descriptors);

	// Define kernel
	typedef linear_kernel<sample_type> lin_kernel;

	// Define the SVM multiclass trainer
	typedef svm_multiclass_linear_trainer <lin_kernel, unsigned long> svm_mc_trainer;
	svm_mc_trainer trainer;

	// Now lets do 5-fold cross-validation
	randomize_samples(samples, labels);
	//phải thử các giá trị của hàm set_c() để xem giá trị nào là phù hợp như ở dưới
		//6 0 0 0 0
		//0 6 0 0 0
		//0 0 6 0 0
		//0 0 0 6 0
		//0 0 0 0 6
	/*
	trainer.set_c(1);
	cout << "cross validation: \n" <<
		cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;
	trainer.set_c(5);
	cout << "cross validation: \n" <<
		cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;
	trainer.set_c(10);
	cout << "cross validation: \n" <<
		cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;
	trainer.set_c(50);
	cout << "cross validation: \n" <<
		cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;
	*/
	trainer.set_c(100);// it good!
	// cout << "cross validation: \n" <<
	// 	cross_validate_multiclass_trainer(trainer, samples, labels, 9) << endl;

	// Train and obtain the decision rule
	multiclass_linear_decision_function<lin_kernel, unsigned long> df = trainer.train(samples, labels);
	serialize("svm.dat") << df;
	/*
	cout << "predicted label: " << df(samples[0]) << ", true label: " << labels[0] << endl;
	cout << "predicted label: " << df(samples[9]) << ", true label: " << labels[9] << endl;
	cout << "predicted label: " << df(samples[19]) << ", true label: " << labels[19] << endl;
	cout << "predicted label: " << df(samples[27]) << ", true label: " << labels[27] << endl;
	cout << "predicted label: " << df(samples[36]) << ", true label: " << labels[36] << endl;
	cout << "predicted label: " << df(samples[44]) << ", true label: " << labels[44] << endl;
	cout << "predicted label: " << df(samples[4]) << ", true label: " << labels[4] << endl;
	cout << "predicted label: " << df(samples[12]) << ", true label: " << labels[12] << endl;
	*/
}

// ---------------------------


//--------------------------------------------
//mạng dnn cho ra vector 1x128, còn svm chì nhận vector cột nên đưa về dạng 128x1
void generate_data(
	std::vector<sample_type>& samples,
	std::vector<matrix<float, 0, 1>> data_face_descriptors)
{
	matrix<double, 128, 1> tmp;

	for (int i = 0; i < data_face_descriptors.size(); i++) {
		for (int j = 0; j < 128; j++) {
			tmp(j) = (double)data_face_descriptors[i](j);
		}
		samples.push_back(tmp);
	}
}
//--------------------------------------------

void predict(std::vector<matrix<float, 0, 1>> face_descriptors)
{
	std::vector<sample_type> samples;
	generate_data(samples, face_descriptors);

	typedef linear_kernel<sample_type> lin_kernel;
	multiclass_linear_decision_function<lin_kernel, unsigned long> df;
	deserialize("svm.dat") >> df;

	cout << "predict label: " << df(samples[0]) << endl;
}