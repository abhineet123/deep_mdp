#ifdef _WIN32
#define hypot _hypot
#endif

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "math.h"
#include <limits>
#include <vector>

#ifdef _CHAR16T
#define CHAR16_T
#endif
#include <Python.h>
#include <patchlevel.h>
#include <numpy/arrayobject.h>

#ifndef MTF_NOT_AVAILABLE
#include "mtf/Utilities/miscUtils.h"
#endif

static std::vector<cv::Point2f> points[3];

static PyArrayObject *img_py_1, *img_py_2;
static PyArrayObject *pts_py_1, *pts_py_2;
static int lk_level;
static int lk_win_size = 4;
static int lk_n_iters = 20;
static int ncc_win_size = 10;
static int pause_after_frame = 1;
static int show_points;

static PyObject* initialize(PyObject* self, PyObject* args);
static PyObject* get(PyObject* self, PyObject* args);

static PyMethodDef pyLKGPUMethods[] = {
	{ "initialize", initialize, METH_VARARGS },
	{ "get", get, METH_VARARGS },
	{ NULL, NULL }     /* Sentinel - marks the end of this structure */
};



#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initpyLKGPU() {
	(void)Py_InitModule("pyLKGPU", pyLKGPUMethods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}
#else
static struct PyModuleDef pyLKGPUModule = {
	PyModuleDef_HEAD_INIT,
	"pyLKGPU",   /* name of module */
	NULL, /* module documentation, may be NULL */
	-1,       /* size of per-interpreter state of the module,
			  or -1 if the module keeps state in global variables. */
	pyLKGPUMethods
};
PyMODINIT_FUNC PyInit_pyLKGPU(void) {
	import_array();
	return PyModule_Create(&pyLKGPUModule);
}
#endif

static void euclideanDistance(const std::vector<cv::Point2f> &point1,
	const std::vector<cv::Point2f> &point2, float *match, int nPts) {

	for(int i = 0; i < nPts; i++) {

		match[i] = sqrt((point1[i].x - point2[i].x)*(point1[i].x - point2[i].x) +
			(point1[i].y - point2[i].y)*(point1[i].y - point2[i].y));

	}
}

static void normCrossCorrelation(const cv::Mat &imgI, const cv::Mat &imgJ,
	const std::vector<cv::Point2f> &points0, const std::vector<cv::Point2f> &points1,
	int nPts, const std::vector<uchar> &status, float *match, int winsize, int method) {

	cv::Mat rec0(cv::Size(winsize, winsize), CV_8UC1);
	cv::Mat rec1(cv::Size(winsize, winsize), CV_8UC1);
	cv::Mat res(cv::Size(1, 1), CV_32FC1);

	for(int i = 0; i < nPts; i++) {
		if(status[i] == 1) {
			cv::getRectSubPix(imgI, cv::Size(winsize, winsize), points0[i], rec0);
			cv::getRectSubPix(imgJ, cv::Size(winsize, winsize), points1[i], rec1);
			cv::matchTemplate(rec0, rec1, res, method);
			match[i] = ((float *)(res.data))[0];

		} else {
			match[i] = 0.0;
		}
	}
}
static void download(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec)
{
	vec.resize(d_mat.cols);
	cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
	d_mat.download(mat);
}

static void download(const cv::cuda::GpuMat& d_mat, std::vector<uchar>& vec)
{
	vec.resize(d_mat.cols);
	cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
	d_mat.download(mat);
}

static PyObject* initialize(PyObject* self, PyObject* args) {
	/*parse input*/
	if(!PyArg_ParseTuple(args, "iiiii",
		&lk_level, &lk_win_size, &lk_n_iters, &ncc_win_size, &show_points)) {
		PyErr_SetString(PyExc_IOError, "Input arguments could not be parsed");
		PyErr_PrintEx(1);
	}
	return Py_BuildValue("i", 1);
}

static PyObject* get(PyObject* self, PyObject* args) {

	//PySys_WriteStdout("\nStarting pyLKGPU\n");

	/*parse input*/
	if(!PyArg_ParseTuple(args, "O!O!O!O!",
		&PyArray_Type, &img_py_1,
		&PyArray_Type, &img_py_2,
		&PyArray_Type, &pts_py_1,
		&PyArray_Type, &pts_py_2)) {
		PyErr_SetString(PyExc_IOError, "Input arguments could not be parsed");
		PyErr_PrintEx(1);
	}
	if(img_py_1 == NULL || img_py_2 == NULL) {
		PyErr_SetString(PyExc_IOError, "img_py_1 or img_py_2 is NULL");
		PyErr_PrintEx(1);
	}
	if(pts_py_1 == NULL || pts_py_2 == NULL) {
		PyErr_SetString(PyExc_IOError, "pts_py_1 or pts_py_2 is NULL");
		PyErr_PrintEx(1);
	}
	if(img_py_1->nd != 2 || img_py_2->nd != 2) {
		PyErr_SetString(PyExc_IOError, "Both input images must be 2 dimensional arrays");
		PyErr_PrintEx(1);
	}
	if(pts_py_1->nd != 2 || pts_py_2->nd != 2) {
		PyErr_SetString(PyExc_IOError, "Both point sets must be 2 dimensional arrays");
		PyErr_PrintEx(1);
	}

	int img_height = img_py_1->dimensions[0];
	int img_width = img_py_1->dimensions[1];
	if(img_height != img_py_2->dimensions[0] || img_width != img_py_2->dimensions[1]) {
		PyErr_SetString(PyExc_IOError, "Input images have inconsistent dimensions");
		PyErr_PrintEx(1);
	}
	if(pts_py_1->dimensions[1] != 2 || pts_py_2->dimensions[1] != 2) {
		PyErr_SetString(PyExc_IOError, "Both point sets must have 2 columns");
		PyErr_PrintEx(1);
	}
	int nPts = pts_py_1->dimensions[0];
	if(nPts != pts_py_2->dimensions[0]) {
		PyErr_SetString(PyExc_IOError, "Both point sets must have the same number of points");
		PyErr_PrintEx(1);
	}
	//PySys_WriteStdout("img_height: %d\t img_width: %d\n", img_height, img_width);
	//PySys_WriteStdout("nPts: %d\n", nPts);
	//PySys_WriteStdout("Level: %d\n", Level);

	//int dummy_input;
	//scanf("Press any key to continue: %d", &dummy_input);

	// Images
	cv::Mat img_1(img_height, img_width, CV_8UC1, img_py_1->data);
	cv::Mat img_2(img_height, img_width, CV_8UC1, img_py_2->data);
	// Points	
	cv::Mat ptsI = cv::Mat(nPts, 2, CV_64FC1, pts_py_1->data);
	cv::Mat ptsJ = cv::Mat(nPts, 2, CV_64FC1, pts_py_2->data);

	points[0].resize(nPts);
	points[1].resize(nPts);
	points[2].resize(nPts);
	for(int i = 0; i < nPts; i++) {
		points[0][i].x = ptsI.at<double>(i, 0); points[0][i].y = ptsI.at<double>(i, 1);
		points[1][i].x = ptsJ.at<double>(i, 0); points[1][i].y = ptsJ.at<double>(i, 1);
		points[2][i].x = ptsI.at<double>(i, 0); points[2][i].y = ptsI.at<double>(i, 1);
	}
	cv::Mat prevPts(1, nPts, CV_32FC2, (void*)&points[0][0]);
	cv::Mat nextPts(1, nPts, CV_32FC2, (void*)&points[1][0]);

	cv::cuda::GpuMat img_1_gpu(img_1);
	cv::cuda::GpuMat img_2_gpu(img_2);
	cv::cuda::GpuMat d_prevPts(prevPts);
	cv::cuda::GpuMat d_nextPts(nextPts);
	cv::cuda::GpuMat d_prevPts2(prevPts);
	cv::cuda::GpuMat d_status;

	cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
		cv::Size(lk_win_size, lk_win_size), lk_level, lk_n_iters, true);
	d_pyrLK_sparse->calc(img_1_gpu, img_2_gpu, d_prevPts, d_nextPts, d_status);
	//PySys_WriteStdout("Completed forward flow\n");
	d_pyrLK_sparse->calc(img_2_gpu, img_1_gpu, d_nextPts, d_prevPts2, d_status);
	//PySys_WriteStdout("Completed backward flow\n");

	download(d_prevPts, points[0]);
	download(d_nextPts, points[1]);
	download(d_prevPts2, points[2]);
	std::vector<uchar> status(d_status.cols);
	download(d_status, status);

	float *fb = (float*)cvAlloc(nPts*sizeof(float));
	float *ncc = (float*)cvAlloc(nPts*sizeof(float));
	normCrossCorrelation(img_1, img_2, points[0], points[1], nPts, status, ncc, ncc_win_size, CV_TM_CCOEFF_NORMED);
	//PySys_WriteStdout("Completed NCC computation \n");

	//float *ssd = (float*)cvAlloc(nPts*sizeof(float));
	//normCrossCorrelation(IMG[I],IMG[J],points[0],points[1],nPts, status, ssd, Winsize,CV_TM_SQDIFF);

	euclideanDistance(points[0], points[2], fb, nPts);
	//PySys_WriteStdout("Completed Euclidean distance computation\n");

	// Output
	int dims[] = { nPts, 4 };
	PyArrayObject *output_py = (PyArrayObject *)PyArray_FromDims(2, dims, NPY_DOUBLE);
	cv::Mat output = cv::Mat(nPts, 4, CV_64FC1, output_py->data);
	double nan = std::numeric_limits<double>::quiet_NaN();
	//float inf = std::numeric_limits<float>::infinity();
	for(int i = 0; i < nPts; i++) {
		if(status[i] == 1) {
			output.at<double>(i, 0) = (double)points[1][i].x;
			output.at<double>(i, 1) = (double)points[1][i].y;
			output.at<double>(i, 2) = (double)fb[i];
			output.at<double>(i, 3) = (double)ncc[i];
		} else {
			output.at<double>(i, 0) = nan;
			output.at<double>(i, 1) = nan;
			output.at<double>(i, 2) = nan;
			output.at<double>(i, 3) = nan;
		}
	}
	//PySys_WriteStdout("Completed writing to output matrix\n");

#ifndef MTF_NOT_AVAILABLE
	if(show_points) {
		std::vector<cv::Mat> img_list;
		img_list.push_back(img_1.clone());
		img_list.push_back(img_2.clone());
		mtf::utils::drawPts<double>(img_list[0], ptsI, cv::Scalar(0, 0, 0), 2);
		mtf::utils::drawPts<double>(img_list[1], ptsJ, cv::Scalar(0, 0, 0), 2);
		//cv::Mat stacked_img = mtf::utils::stackImages(img_list);
		cv::imshow("pyLKGPU::Input Image 1", img_list[0]);
		cv::imshow("pyLKGPU::Input Image 2", img_list[1]);
		//mtf::utils::printMatrixToFile<float>(img_1, nullptr, "log/img_1.txt", "%d");
		//mtf::utils::printMatrixToFile<float>(img_2, nullptr, "log/img_2.txt", "%d");
		mtf::utils::printMatrixToFile<double>(ptsI, nullptr, "log/ptsI.txt", "%.4f");
		mtf::utils::printMatrixToFile<double>(ptsJ, nullptr, "log/ptsJ.txt", "%.4f");
		int key = cv::waitKey(1 - pause_after_frame);
		if(key == 27) {
			Py_Exit(0);
		} else if(key == 32) {
			pause_after_frame = 1 - pause_after_frame;
		}
	}
#endif
	//PySys_WriteStdout("Completed cleaning up\n");
	return Py_BuildValue("O", output_py);
}
