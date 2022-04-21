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

#include <Python.h>
#include <patchlevel.h>
#include <numpy/arrayobject.h>

#include "opencv2/highgui/highgui.hpp"

#include <math.h>

#ifndef NAN
#define NAN 0/0
#endif

#ifndef M_PI
#define M_PI 3.14159265358979L
#endif
// rowwise access
#define coord(x, y, ch, width, height, n_ch) (x*n_ch+y*width+ch)
#define nextrow(tmp, width, height, n_ch) ((tmp)+width*n_ch)
#define nextcol(tmp, width, height, n_ch) ((tmp)+n_ch)
#define nextr_c(tmp, width, height, n_ch) ((tmp)+(width+1)*n_ch)

#define M(r, c) H[r*3+c]

static PyArrayObject *img_py, *H_py, *bb_py;

static PyObject* get(PyObject* self, PyObject* args);

static PyMethodDef pyWarpMethods[] = {
	{ "get", get, METH_VARARGS },
	{ NULL, NULL }     /* Sentinel - marks the end of this structure */
};




#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initpyWarp() {
	(void)Py_InitModule("pyWarp", pyWarpMethods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}
#else
static struct PyModuleDef pyWarpModule = {
	PyModuleDef_HEAD_INIT,
	"pyWarp",   /* name of module */
	NULL, /* module documentation, may be NULL */
	-1,       /* size of per-interpreter state of the module,
			  or -1 if the module keeps state in global variables. */
	pyWarpMethods
};
PyMODINIT_FUNC PyInit_pyWarp(void) {
	import_array();
	return PyModule_Create(&pyWarpModule);
}
#endif

/* Warps image of size w x h, using affine transformation matrix (2x2 part)
   and offset (center of warping) ofsx, ofsy. Result is the region of size
   defined with roi. */
void warp_image_roi(unsigned char *image, int w, int h, int n_ch, float *H,
	float xmin, float xmax, float ymin, float ymax,
	int n_rows, int n_cols, float fill, float *result)
{
	float curx, cury, curz, wx, wy, wz, ox, oy, oz;
	int x, y;
	unsigned char *tmp;
	float *output = result, i, j, xx, yy;
	int ch;
	/* precalulate necessary constant with respect to i,j offset
	   translation, H is column oriented (transposed) */
	ox = M(0, 2);
	oy = M(1, 2);
	oz = M(2, 2);

	yy = ymin;
	for(j = 0; j < n_rows; j++)
	{
		/* calculate x, y for current row */
		curx = M(0, 1)*yy + ox;
		cury = M(1, 1)*yy + oy;
		curz = M(2, 1)*yy + oz;
		xx = xmin;
		yy = yy + 1;
		for(i = 0; i < n_cols; i++)
		{
			/* calculate x, y in current column */
			wx = M(0, 0)*xx + curx;
			wy = M(1, 0)*xx + cury;
			wz = M(2, 0)*xx + curz;
			//       printf("%g %g, %g %g %g\n", xx, yy, wx, wy, wz);
			wx /= wz; wy /= wz;
			xx = xx + 1;

			x = (int)floor(wx);
			y = (int)floor(wy);

			if(x >= 0 && y >= 0)
			{
				wx -= x; wy -= y;
				if(x + 1 == w && wx == 1)
					x--;
				if(y + 1 == h && wy == 1)
					y--;
				if((x + 1) < w && (y + 1) < h)
				{
					for(ch = 0; ch < n_ch; ch++) {
						tmp = &image[coord(x, y, ch, w, h, n_ch)];
						/* image[x,y]*(1-wx)*(1-wy) + image[x+1,y]*wx*(1-wy) +
						image[x,y+1]*(1-wx)*wy + image[x+1,y+1]*wx*wy */
						*output++ =
							(*(tmp)* (1 - wx) + *nextcol(tmp, w, h, n_ch) * wx) * (1 - wy) +
							(*nextrow(tmp, w, h, n_ch) * (1 - wx) + *nextr_c(tmp, w, h, n_ch) * wx) * wy;
					}

				} else
					for(ch = 0; ch < n_ch; ch++) {
						*output++ = fill;
					}
			} else {
				for(ch = 0; ch < n_ch; ch++) {
					*output++ = fill;
				}
			}
		}
	}
}

void to_cv(cv::Mat &result, const float *image, int num_cols, int num_rows)
{
	// convert to matlab's column based representation
	int i, j;
	const float* s_ptr = image;
	float* d_ptr, *data;
	data = (float *)(result.data);
	for(i = 0; i < num_rows; i++)
	{
		d_ptr = &data[i];
		for(j = 0; j < num_cols; j++, d_ptr += num_rows, s_ptr++)
			(*d_ptr) = (*s_ptr);
	}
}

static PyObject* get(PyObject* self, PyObject* args) {
	/*parse first input array*/
	if(!PyArg_ParseTuple(args, "O!O!O!",
		&PyArray_Type, &img_py,
		&PyArray_Type, &H_py,
		&PyArray_Type, &bb_py)) {
		PyErr_SetString(PyExc_IOError, "Input arguments could not be parsed");
		return NULL;
		//PyErr_PrintEx(1);
	}
	if(img_py == NULL || H_py == NULL || bb_py == NULL) {
		PyErr_SetString(PyExc_IOError, "img_py, H_py or bb_py is NULL");
		return NULL;
		//PyErr_PrintEx(1);
	}
	if(H_py->nd != 2) {
		PyErr_SetString(PyExc_IOError, "Warp matrix must be a 2 dimensional array");
		return NULL;
		//PyErr_PrintEx(1);
	}
	if(bb_py->nd != 1) {
		PyErr_SetString(PyExc_IOError, "Bounding box must be a 1 dimensional array");
		return NULL;
		//PyErr_PrintEx(1);
	}

	int h = img_py->dimensions[0];
	int w = img_py->dimensions[1];
	int n_ch = 1;
	if(img_py->nd > 2) {
		n_ch = img_py->dimensions[2];
	}

	unsigned char *im = (unsigned char*)img_py->data;
	float *H = (float*)H_py->data;
	float xmin, xmax, ymin, ymax, fill;
	//from_matlab(prhs[0], &im, &w, &h);
	float *B = (float*)bb_py->data;
	xmin = (*B++); xmax = (*B++);
	ymin = (*B++); ymax = (*B++);
	// Output

	//float n_rows_float = ymax - ymin + 1;
	//float n_cols_float = xmax - xmin + 1;
	//int n_rows = (int)(n_rows_float);
	//int n_cols = (int)(n_cols_float);
	//int n_rows_round = round(n_rows_float);
	//int n_cols_round = round(n_cols_float);
	//printf("ymax: %f ymin: %f xmax: %f xmin: %f\n", ymax, ymin, xmax, xmin);
	//printf("n_rows: %d n_cols: %d\n", n_rows, n_cols);
	//printf("n_rows_round: %d n_cols_round: %d\n", n_rows_round, n_cols_round);
	//printf("n_rows_float: %.30f n_cols_float: %.30f\n", n_rows_float, n_cols_float);

	int _n_rows = (int)(ymax - ymin + 1);
	int _n_cols = (int)(xmax - xmin + 1);

	int dims[] = { _n_rows, _n_cols, n_ch };
	PyArrayObject *output_py = (PyArrayObject *)PyArray_FromDims(3, dims, NPY_FLOAT);
	float *result = (float*)output_py->data;
	fill = 0;
	warp_image_roi(im, w, h, n_ch, H, xmin, xmax, ymin, ymax, _n_rows, _n_cols, fill, result);
	//PySys_WriteStdout("Completed obtaining the warped patch\n");
	return Py_BuildValue("O", output_py);
}
