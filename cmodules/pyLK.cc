#ifdef _WIN32
#define hypot _hypot
#endif

#include "opencv/cv.h"
#include "opencv2/highgui/highgui.hpp"
#if CV_MAJOR_VERSION == 3
#include "opencv2/imgproc/imgproc.hpp"
#endif
#include "math.h"
#include <limits>
#ifdef _CHAR16T
#define CHAR16_T
#endif
#include <Python.h>
#include <patchlevel.h>
#include <numpy/arrayobject.h>

#ifndef MTF_NOT_AVAILABLE
#include "mtf/Utilities/miscUtils.h"
#endif

static const int MAX_COUNT = 500;
static const int MAX_IMG = 2;
static CvPoint2D32f* points[3] = { 0,0,0 };
static PyArrayObject *img_py_1, *img_py_2;
static PyArrayObject *pts_py_1, *pts_py_2;

static int lk_level;
static int lk_win_size = 4;
static int lk_n_iters = 20;
static double lk_eps = 0.03;
static int ncc_win_size = 10;
static int show_points;
static int pause_after_frame = 1;

static PyObject* initialize(PyObject* self, PyObject* args);
static PyObject* get(PyObject* self, PyObject* args);


static PyMethodDef pyLKMethods[] = {
	{ "initialize", initialize, METH_VARARGS },
	{ "get", get, METH_VARARGS },
	{ NULL, NULL }     /* Sentinel - marks the end of this structure */
};

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initpyLK() {
	(void)Py_InitModule("pyLK", pyLKMethods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}
#else
static struct PyModuleDef pyLKModule = {
	PyModuleDef_HEAD_INIT,
	"pyLK",   /* name of module */
	NULL, /* module documentation, may be NULL */
	-1,       /* size of per-interpreter state of the module,
			  or -1 if the module keeps state in global variables. */
	pyLKMethods
};
PyMODINIT_FUNC PyInit_pyLK(void) {
	import_array();
	return PyModule_Create(&pyLKModule);
}
#endif

static void euclideanDistance(CvPoint2D32f *point1, CvPoint2D32f *point2, float *match, int nPts) {

	for(int i = 0; i < nPts; i++) {

		match[i] = sqrt((point1[i].x - point2[i].x)*(point1[i].x - point2[i].x) +
			(point1[i].y - point2[i].y)*(point1[i].y - point2[i].y));

	}
}

static void normCrossCorrelation(IplImage *imgI, IplImage *imgJ, CvPoint2D32f *points0, CvPoint2D32f *points1,
	int nPts, char *status, float *match, int winsize, int method) {


	IplImage *rec0 = cvCreateImage(cvSize(winsize, winsize), 8, 1);
	IplImage *rec1 = cvCreateImage(cvSize(winsize, winsize), 8, 1);
	IplImage *res = cvCreateImage(cvSize(1, 1), IPL_DEPTH_32F, 1);

	for(int i = 0; i < nPts; i++) {
		if(status[i] == 1) {
			cvGetRectSubPix(imgI, rec0, points0[i]);
			cvGetRectSubPix(imgJ, rec1, points1[i]);
			cvMatchTemplate(rec0, rec1, res, method);
			match[i] = ((float *)(res->imageData))[0];

		} else {
			match[i] = 0.0;
		}
	}
	cvReleaseImage(&rec0);
	cvReleaseImage(&rec1);
	cvReleaseImage(&res);

}
static PyObject* initialize(PyObject* self, PyObject* args) {
	/*parse input*/
	if(!PyArg_ParseTuple(args, "iiidii",
		&lk_level, &lk_win_size, &lk_n_iters, &lk_eps, &ncc_win_size, &show_points)) {
		PyErr_SetString(PyExc_IOError, "Input arguments could not be parsed");
		PyErr_PrintEx(1);
	}
	return Py_BuildValue("i", 1);
}


template<typename ScalarT>
void drawPts(cv::Mat &img, const cv::Mat &pts, cv::Scalar col, int radius = 2,
	int thickness = -1) {
	for (int pt_id = 0; pt_id < pts.rows; ++pt_id) {
		cv::Point pt(int(pts.at<ScalarT>(pt_id, 0)), int(pts.at<ScalarT>(pt_id, 1)));
		cv::circle(img, pt, radius, col, thickness);
	}
}


static PyObject* get(PyObject* self, PyObject* args) {

	//PySys_WriteStdout("\nStarting pyLK\n");

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

	IplImage **IMG = (IplImage**)calloc(MAX_IMG, sizeof(IplImage*));
	IplImage **PYR = (IplImage**)calloc(MAX_IMG, sizeof(IplImage*));

	int I = 0;
	int J = 1;

	// Images
	cv::Mat img_1(img_height, img_width, CV_8UC1, img_py_1->data);
	cv::Mat img_2(img_height, img_width, CV_8UC1, img_py_2->data);

	CvSize imageSize = cvSize(img_width, img_height);
	IMG[I] = new IplImage(img_1);
	PYR[I] = cvCreateImage(imageSize, 8, 1);
	IMG[J] = new IplImage(img_2);
	PYR[J] = cvCreateImage(imageSize, 8, 1);

	// Points	
	cv::Mat ptsI = cv::Mat(nPts, 2, CV_64FC1, pts_py_1->data);
	cv::Mat ptsJ = cv::Mat(nPts, 2, CV_64FC1, pts_py_2->data);

	points[0] = (CvPoint2D32f*)cvAlloc(nPts*sizeof(CvPoint2D32f)); // template
	points[1] = (CvPoint2D32f*)cvAlloc(nPts*sizeof(CvPoint2D32f)); // target
	points[2] = (CvPoint2D32f*)cvAlloc(nPts*sizeof(CvPoint2D32f)); // forward-backward

	for(int i = 0; i < nPts; i++) {
		points[0][i].x = ptsI.at<double>(i, 0); points[0][i].y = ptsI.at<double>(i, 1);
		points[1][i].x = ptsJ.at<double>(i, 0); points[1][i].y = ptsJ.at<double>(i, 1);
		points[2][i].x = ptsI.at<double>(i, 0); points[2][i].y = ptsI.at<double>(i, 1);
	}
	float *fb = (float*)cvAlloc(nPts*sizeof(float));
	char  *status = (char*)cvAlloc(nPts);
	cvCalcOpticalFlowPyrLK(IMG[I], IMG[J], PYR[I], PYR[J], points[0],
		points[1], nPts, cvSize(lk_win_size, lk_win_size), lk_level, status, 0,
		cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, lk_n_iters, lk_eps), CV_LKFLOW_INITIAL_GUESSES);
	//PySys_WriteStdout("Completed forward flow\n");
	cvCalcOpticalFlowPyrLK(IMG[J], IMG[I], PYR[J], PYR[I], points[1],
		points[2], nPts, cvSize(lk_win_size, lk_win_size), lk_level, status, 0,
		cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, lk_n_iters, lk_eps),
		CV_LKFLOW_INITIAL_GUESSES | CV_LKFLOW_PYR_A_READY | CV_LKFLOW_PYR_B_READY);
	//PySys_WriteStdout("Completed backward flow\n");

	float *ncc = (float*)cvAlloc(nPts*sizeof(float));
	normCrossCorrelation(IMG[I], IMG[J], points[0], points[1], nPts, status, ncc, ncc_win_size, CV_TM_CCOEFF_NORMED);
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
	//show_points = 1;
//#ifndef MTF_NOT_AVAILABLE
	if(show_points) {
		std::vector<cv::Mat> img_list;
		img_list.push_back(img_1.clone());
		img_list.push_back(img_2.clone());
		drawPts<double>(img_list[0], ptsI, cv::Scalar(0, 0, 0), 2);
		drawPts<double>(img_list[1], ptsJ, cv::Scalar(0, 0, 0), 2);
		//cv::Mat stacked_img = mtf::utils::stackImages(img_list);
		cv::imshow("pyLK::Input Image 1", img_list[0]);
		cv::imshow("pyLK::Input Image 2", img_list[1]);
		//mtf::utils::printMatrixToFile<float>(img_1, nullptr, "log/img_1.txt", "%d");
		//mtf::utils::printMatrixToFile<float>(img_2, nullptr, "log/img_2.txt", "%d");
		//mtf::utils::printMatrixToFile<double>(ptsI, nullptr, "log/ptsI.txt", "%.4f");
		//mtf::utils::printMatrixToFile<double>(ptsJ, nullptr, "log/ptsJ.txt", "%.4f");
		int key = cv::waitKey(1 - pause_after_frame);
		if(key == 27) {
			Py_Exit(0);
		} else if(key == 32) {
			pause_after_frame = 1 - pause_after_frame;
		}
	}
//#endif

	// clean up
	for(int i = 0; i < MAX_IMG; i++) {
		//! sharing memory with Python array
		//cvReleaseImage(&(IMG[i]));
		cvReleaseImage(&(PYR[i]));
	}
	free(IMG);
	free(PYR);
	//PySys_WriteStdout("Completed cleaning up\n");
	return Py_BuildValue("O", output_py);
}


