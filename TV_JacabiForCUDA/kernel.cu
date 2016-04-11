#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cuda.h>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
#include <cmath>
#include <sm_11_atomic_functions.h>
#include <sm_20_atomic_functions.h>
using namespace std;
using namespace cv;

/*
double** newDoubleMatrix(int nx, int ny)
{
	double** matrix = new double*[ny];

	for (int i = 0; i < ny; i++)
	{
		matrix[i] = new double[nx];
	}
	if (!matrix)
		return NULL;
	return
		matrix;
}
bool deleteDoubleMatrix(double** matrix, int nx, int ny)
{
	if (!matrix)
	{
		return true;
	}
	for (int i = 0; i < ny; i++)
	{
		if (matrix[i])
		{
			delete[] matrix[i];
		}
	}
	delete[] matrix;

	return true;
}
*/

__global__ void TV(unsigned char* buffer,long size,unsigned int* histo)
{

}

Mat TVDenoising(Mat img, int iter)
{
	int ep = 1;
	int nx = img.cols*3;
	int ny = img.rows;
	double dt = (double)ep / 5.0f;
	//double lam = 0.0;
	double lam = 0.04;
	int ep2 = ep*ep;

	uchar *data=img.data;
	double** image;
	double** image0;
	double** imageaf;

	cudaMalloc((void **)&image,nx*ny);
	cudaMalloc((void **)&image0,nx*ny);
	cudaMalloc((void **)&imageaf,nx*ny);
	cudaMemcpy(image,data,nx*ny,cudaMemcpyHostToDevice);
	cudaMemcpy(image0,data,nx*ny,cudaMemcpyHostToDevice);
	cudaMemcpy(imageaf,data,nx*ny,cudaMemcpyHostToDevice);

	//patr1_CUDA
	/*
	for (int i = 0; i<ny; i++)
	{
		uchar* p = img.ptr<uchar>(i);
		for (int j = 0; j<nx; j++)
		{
			imageaf[i][j] = image0[i][j] = image[i][j] = (double)p[j];
		}
	}
	*/

	for (int t = 1; t <= iter; t++)
	{
		//part2_CUDA
		for (int i = 0; i < ny; i++)
		{
			for (int j = 0; j < nx; j++)
			{
				int tmp_i1 = (i + 1)<ny ? (i + 1) : (ny - 1);
				int tmp_j1 = (j + 1)<nx ? (j + 1) : (nx - 1);
				int tmp_i2 = (i - 1) > -1 ? (i - 1) : 0;
				int tmp_j2 = (j - 1) > -1 ? (j - 1) : 0;

				double tmp_x = (image[i][tmp_j1] - image[i][tmp_j2]) / 2;
				double tmp_y = (image[tmp_i1][j] - image[tmp_i2][j]) / 2;
				double tmp_xx = image[i][tmp_j1] + image[i][tmp_j2] - image[i][j] * 2; 
				double tmp_yy = image[tmp_i1][j] + image[tmp_i2][j] - image[i][j] * 2; 
				double tmp_dp = image[tmp_i1][tmp_j1] + image[tmp_i2][tmp_j2]; 
				double tmp_dm = image[tmp_i2][tmp_j1] + image[tmp_i1][tmp_j2];
				double tmp_xy = (tmp_dp - tmp_dm) / 4;
				double tmp_num = tmp_xx*(tmp_y*tmp_y + ep2)
					- 2 * tmp_x*tmp_y*tmp_xy + tmp_yy*(tmp_x*tmp_x + ep2); 
				double tmp_den = pow((tmp_x*tmp_x + tmp_y*tmp_y + ep2), 1.5);

				imageaf[i][j] += dt*tmp_num / tmp_den + dt*lam*(image0[i][j] - image[i][j]);
				//imageaf[i][j] += dt*lam*tmp_num / tmp_den + dt*(image0[i][j] - image[i][j]);
			}
		}

		for (int i = 0; i < ny; i++)
		for (int j = 0; j < nx; j++)
			image[i][j]=imageaf[i][j];
	}

	Mat new_img;
	uchar *new_data=img.data;

	img.copyTo(new_img);

	//part3_CUDA
	/*
	for (int i = 0; i<ny; i++)
	{
		//uchar* p = img.ptr<uchar>(i);
		uchar* np = new_img.ptr<uchar>(i);
		for (int j = 0; j<nx; j++){
			int tmp = (int)imageaf[i][j];
			tmp = max(0, min(tmp, 255));
			np[j] = (uchar)(tmp);
		}
	}
	*/

	//deleteDoubleMatrix(image0, nx, ny);
	//deleteDoubleMatrix(image, nx, ny);
	//deleteDoubleMatrix(imageaf,nx,ny);

	cudaFree(image);
	cudaFree(image0);
	cudaFree(imageaf);

	return new_img;
}

int main(void)
{
	string s;
	int t;
	Mat src,after;
	double dt,lam;

	cout<<"Address:\n";
	cin>>s;

	cout<<"Time:\n";
	cin>>t;

	cout<<"What is your dt?lam?\n";
	cin>>dt>>lam;

	src=imread(s.c_str());

	after=TVDenoising(src,t);

	imshow("Before",src);
	imshow("After",after);
	waitKey(0);

	return 0;
}