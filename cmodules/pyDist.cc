// Copyright 2011 Zdenek Kalal
//
// This file is part of TLD.
// 
// TLD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// TLD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with TLD.  If not, see <http://www.gnu.org/licenses/>.

#include <stdio.h>
#include "math.h"

#include <Python.h>
#include <patchlevel.h>
#include <numpy/arrayobject.h>

static PyArrayObject *img_py_1, *img_py_2;

static PyObject* get(PyObject* self, PyObject* args);

static PyMethodDef pyDistMethods[] = {
	{ "get", get, METH_VARARGS },
	{ NULL, NULL }     /* Sentinel - marks the end of this structure */
};



#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initpyDist() {
	(void)Py_InitModule("pyDist", pyDistMethods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}
#else
static struct PyModuleDef pyDistModule = {
	PyModuleDef_HEAD_INIT,
	"pyDist",   /* name of module */
	NULL, /* module documentation, may be NULL */
	-1,       /* size of per-interpreter state of the module,
			  or -1 if the module keeps state in global variables. */
	pyDistMethods
};
PyMODINIT_FUNC PyInit_pyDist(void) {
	import_array();
	return PyModule_Create(&pyDistModule);
}
#endif

// correlation
float ccorr(float *f1,float *f2,int numDim) {
	float f = 0;
	for (int i = 0; i<numDim; i++) {
		f += f1[i]*f2[i];
	}
	return f;
}

// correlation normalized
float ccorr_normed(float *f1, float *f2,int numDim) {
	float corr = 0;
	float norm1 = 0;
	float norm2 = 0;

	for (int i = 0; i<numDim; i++) {
		corr += f1[i]*f2[i];
		norm1 += f1[i]*f1[i];
		norm2 += f2[i]*f2[i];
	}
	// normalization to <0,1>
	return (corr / sqrt(norm1*norm2) + 1) / 2.0;
}

// euclidean distance
float euclidean(float *f1, float *f2,int numDim) {

	float sum = 0;
	for (int i = 0; i<numDim; i++) {
		sum += (f1[i]-f2[i])*(f1[i]-f2[i]);
	}
	return sqrt(sum);
}


static PyObject* get(PyObject* self, PyObject* args) {
	int flag;
	/*parse first input array*/
	if(!PyArg_ParseTuple(args, "O!O!|i",
		&PyArray_Type, &img_py_1,
		&PyArray_Type, &img_py_2, &flag)) {
		PyErr_SetString(PyExc_IOError, "Input arguments could not be parsed");
		PyErr_PrintEx(1);
	}
	if(img_py_1 == NULL || img_py_2 == NULL) {
		PyErr_SetString(PyExc_IOError, "img_py_1 or img_py_2 is NULL");
		PyErr_PrintEx(1);
	}
	if(img_py_1->nd != 2 || img_py_2->nd != 2) {
		PyErr_SetString(PyExc_IOError, "Both input images must be 2 dimensional arrays");
		PyErr_PrintEx(1);
	}
	float *x1 = (float*)img_py_1->data;
	int N1 = img_py_1->dimensions[0];
	int M1 = img_py_1->dimensions[1];
	float *x2 = (float*)img_py_2->data;
	int N2 = img_py_2->dimensions[0];
	int M2 = img_py_2->dimensions[1];

	//PySys_WriteStdout("M1: %d M2: %d\n", M1, M2);
	//PySys_WriteStdout("N1: %d N2: %d\n", N1, N2);

	if (M1 != M2) {
		PyErr_SetString(PyExc_IOError, "Both sets of input images must have the same size");
		PyErr_PrintEx(1);
	}
	int dims[] = { N2, N1 };
	PyArrayObject *output_py = (PyArrayObject *)PyArray_FromDims(2, dims, NPY_FLOAT);
	float *resp = (float*)output_py->data;

	switch (flag)
	{
	case 1 :
		for (int i = 0; i < N2; i++) {
			for (int ii = 0; ii < N1; ii++) {
				*resp++ = ccorr_normed(x1+ii*M1,x2+i*M1,M1);
			}
		}

		break;
	case 2 :

		for (int i = 0; i < N2; i++) {
			for (int ii = 0; ii < N1; ii++) {
				*resp++ = euclidean(x1+ii*M1,x2+i*M1,M1);
			}
		}

		break;
	}
	//PySys_WriteStdout("Completed computing distance\n");
	return Py_BuildValue("O", output_py);
}
