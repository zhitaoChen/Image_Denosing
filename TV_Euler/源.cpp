#include <iostream> 
#include <cstdio>
#include <cstdlib>  
#include <cmath>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
using namespace std;
using namespace cv;

double **newDoubleMatrix(int x,int y)
{
	double** a=new double *[x];

	for(int i=0;i<x;i++)
		a[i]=new double[y];

	//memset(a,0,sizeof(a));
	return a;
}

double eps=1e-9;
double C(double a,double b,double c,double d)
{
	return sqrt(pow(a-b,2)+pow((c-d)/2,2)+pow(eps,2));
}

Mat TVDenoising(Mat img, int iter,double dt,double lam)
{
	int nx=3*img.cols;
	int ny=img.rows;

	/*
	int ep = 1;
	double dt = 0.25f;
	double lam = 0.0;
	int ep2 = ep*ep;
	*/

	double** image_be = newDoubleMatrix(ny, nx);//before iter
	double** image_af = newDoubleMatrix(ny, nx);//after iter
	double** image_src = newDoubleMatrix(ny,nx);//source

	for(int i=0;i<ny;i++)
	{
		uchar* p=img.ptr<uchar>(i);

		for(int j=0;j<nx;j++)
			image_src[i][j]=image_be[i][j]=image_af[i][j]=(double)p[j];
	}

	for (int t = 1; t <= iter; t++)
	{
		for (int i = 0; i < ny; i++)
		{
			for (int j = 0; j < nx; j++)
			{
				int iadd1=(i+1)<ny ? (i+1) :(ny-1);
				int jadd1=(j+1)<nx ? (j+1): (nx-1);
				int isub1=(i-1) > -1 ? (i-1) : 0;
				int jsub1=(j-1) > -1 ? (j-1) : 0;

				double c1=C(image_be[iadd1][j],image_be[i][j],image_be[i][jadd1],image_be[i][jsub1]);
				double c2=C(image_be[i][j],image_be[isub1][j],image_be[isub1][jadd1],image_be[isub1][jsub1]);
				double c3=C(image_be[i][jadd1],image_be[i][j],image_be[iadd1][j],image_be[isub1][j]);
				double c4=C(image_be[i][j],image_be[i][jsub1],image_be[iadd1][jsub1],image_be[isub1][jsub1]);
				double c=c1+c2+c3+c4;

				image_af[i][j]=(image_src[i][j]+lam*(c1*image_be[iadd1][j]+c2*image_be[isub1][j]+c3*image_be[i][jadd1]+c4*image_be[i][jsub1])
					-(lam*c*image_be[i][j]));
			}
		}

		for(int i=0;i<ny;i++)
		for(int j=0;j<nx;j++)
			image_be[i][j]=image_af[i][j];
	}

	Mat new_img=img.clone();

	for(int i=0;i<ny;i++)
	for(int j=0;j<nx;j++)
	{
		int x=img.at<uchar>(i,j);
		int y=(int)image_af[i][j];
		if(x==y)
			printf("%d %d\n",i,j);

		new_img.at<uchar>(i,j)=(uchar)image_af[i][j];
	}

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

	after=TVDenoising(src,t,dt,lam);

	imshow("Before",src);
	imshow("After",after);
	waitKey(0);

	return 0;
}