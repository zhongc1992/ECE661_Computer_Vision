
#include "cv.h"  
#include "highgui.h"   
#include <ml.h>  
#include <iostream>  
#include <fstream>  
#include <string>  
#include <vector>  
using namespace cv;  
using namespace std;  

void cross_product(Mat p1, Mat p2, Mat &result); //p1 x p2 = result
void vanish(Mat P, Mat Q, Mat R, Mat S, Mat& vanish);//PQRS points and vanishing line
void affine_homo(Mat PQ,Mat PR,Mat SQ,Mat SR,Mat &H);
void onestep_homo(Mat l1_x,Mat l1_y,Mat l2_x,Mat l2_y,Mat l3_x,Mat l3_y,Mat l4_x,Mat l4_y,Mat l5_x, Mat l5_y,Mat &H);
void mat_helper(Mat l1_x,Mat l1_y, int row, Mat &large, Mat &small);
float max(float num1, float num2);
float min(float num1, float num2);
void bilinear(float col, float row ,float x0,float y0,Mat source, Mat &dest);
void Map_homo(float height, float width, Point2f P_i, Point2f P_f, Point2f S_f, Mat face, Mat &target);

/*******************************************************************
a helper function helps to build 5x5 and 5x1 matrix used to calculate in one-step method
*******************************************************************/
void mat_helper(Mat l1_x,Mat l1_y, int row, Mat &large, Mat &small) {
	float l1 = l1_x.at<float>(0,0)/l1_x.at<float>(2,0);
	float l2 = l1_x.at<float>(1,0)/l1_x.at<float>(2,0);
	float l3 = l1_x.at<float>(2,0)/l1_x.at<float>(2,0);
	float m1 = l1_y.at<float>(0,0)/l1_y.at<float>(2,0);
	float m2 = l1_y.at<float>(1,0)/l1_y.at<float>(2,0);
	float m3 = l1_y.at<float>(2,0)/l1_y.at<float>(2,0);
	large.at<float>(row,0) = l1*m1;
	large.at<float>(row,1) = (l1*m2 + l2*m1)/2;
	large.at<float>(row,2) = l2*m2;
	large.at<float>(row,3) = (l1*m3 + l3*m1)/2;
	large.at<float>(row,4) = (l2*m3 + l3*m2)/2;
	small.at<float>(row,0) = -l3 * m3;
}
/**************************************************************
calculate homography for one-step method
**************************************************************/
void onestep_homo(Mat l1_x,Mat l1_y,Mat l2_x,Mat l2_y,Mat l3_x,Mat l3_y,Mat l4_x,Mat l4_y,Mat l5_x, Mat l5_y,Mat &H) {
	Mat large = Mat (5,5,CV_32F); //matrix on the left side
	Mat small = Mat (5,1,CV_32F); //matrix on the right side
	Mat right_hand = Mat (1,2,CV_32F); //matrix to solve V_spe^T * A^T = right_hand

	Mat V, D2, D, Vt,A,V_spe,temp;

	mat_helper(l1_x,l1_y, 0, large, small); //a help function to build large and small matrix
	mat_helper(l2_x,l2_y, 1, large, small);
	mat_helper(l3_x,l3_y, 2, large, small);
	mat_helper(l4_x,l4_y, 3, large, small);
	mat_helper(l5_x,l5_y, 4, large, small);
	cout <<" Large is " <<endl<<large<<endl<<endl;
	cout <<" Small is" <<endl<<small<<endl<<endl;
	//Mat temp = large.inv(DECOMP_SVD) * small;
	solve(large,small,temp);
	//cout<<"The temp value.."<<endl<<temp<<endl<<endl;
	float a = temp.at<float>(0,0);
	float b = temp.at<float>(1,0);
	float c = temp.at<float>(2,0);
	float d = temp.at<float>(3,0);
	float e = temp.at<float>(4,0);
	Mat S = Mat (2,2,CV_32F);
	S.at<float>(0,0) = a;
	S.at<float>(0,1) = b/2;
	S.at<float>(1,0) = b/2;
	S.at<float>(1,1) = c;

	right_hand.at<float>(0,0) = d/2;
	right_hand.at<float>(0,1) = e/2;
	SVD all(S,SVD::FULL_UV);
	V = all.u;
	D2 = all.w;
	Vt = all.vt;
	cout<<" S value is "<<endl<<S<<endl<<endl;
	cout<<" V value is "<<endl<<V<<endl<<endl;
	cout<<" Vt value is "<<endl<<Vt<<endl<<endl;
	sqrt(D2, D);
	D = Mat::diag(D);
	A = V*D*Vt;
	cout<<" A value is "<<endl<<A<<endl<<endl;
	Mat A_inv = A.inv();
	V_spe  = right_hand * A_inv.t(); //the bottom two values of H
	cout<<" V_spe value is "<<endl<<V_spe<<endl<<endl;

	H.at<float>(0,0) = A.at<float>(0,0)/A.at<float>(0,0);
	H.at<float>(0,1) = A.at<float>(0,1)/A.at<float>(0,0);
	H.at<float>(1,0) = A.at<float>(1,0)/A.at<float>(0,0);
	H.at<float>(1,1) = A.at<float>(1,1)/A.at<float>(0,0);
	H.at<float>(2,0) = V_spe.at<float>(0,0)/A.at<float>(0,0);
	H.at<float>(2,1) = V_spe.at<float>(0,1)/A.at<float>(0,0);
	H.at<float>(2,2) = 0.5;//adjust the scale from 1 to 0.5
	cout<<" H value is "<<endl<<H<<endl<<endl;
}

/***********************************************************************
Calculate the affine homography based on four points-- 2 orthogonal pair line
***********************************************************************/
void affine_homo(Mat PQ,Mat PR,Mat SQ,Mat SR,Mat &H) {
	Mat large = Mat (2,2,CV_32F);//matrix on the left hand of equation
	Mat small = Mat (2,1,CV_32F);//matrix on the right hand of equation
	Mat V, D2, D, Vt; // S = V*D2*Vt
	Mat S = Mat (2,2,CV_32F);//matrix S
	Mat A;//matrix A

	float l1 = PQ.at<float>(0,0);
	float l2 = PQ.at<float>(1,0);
	float m1 = PR.at<float>(0,0);
	float m2 = PR.at<float>(1,0);
	large.at<float>(0,0) = l1*m1;
	large.at<float>(0,1) = l1*m2 + l2*m1;
	small.at<float>(0,0) = -l2*m2;

	l1 = SQ.at<float>(0,0);
	l2 = SQ.at<float>(1,0);
	m1 = SR.at<float>(0,0);
	m2 = SR.at<float>(1,0);
	large.at<float>(1,0) = l1*m1;
	large.at<float>(1,1) = l1*m2 + l2*m1;
	small.at<float>(1,0) = -l2*m2;

	Mat temp = large.inv(DECOMP_SVD) * small;

	float s11 = temp.at<float>(0,0);
	float s12 = temp.at<float>(1,0);
	float s22 = 1;

	S.at<float>(0,0) = s11;
	S.at<float>(0,1) = s12;
	S.at<float>(1,0) = s12;
	S.at<float>(1,1) = s22;
	cout << "S is "<<endl<<S <<endl<<endl;

	SVD all(S,SVD::FULL_UV);
	V = all.u;
	D2 = all.w;
	Vt = all.vt;
	sqrt(D2, D);
	D = Mat::diag(D);
	//cout <<"This is D " <<endl<<D<<endl<<endl;
	//cout<<"This is V "<<endl<<V<<endl<<endl;
	//cout<<"This is Vt "<<endl<<Vt<<endl<<endl;
	//A = V*D*Vt;
	A = V*D*Vt;
	H.at<float>(0,0) = A.at<float>(0,0);
	H.at<float>(0,1) = A.at<float>(0,1);
	H.at<float>(1,0) = A.at<float>(1,0);
	H.at<float>(1,1) = A.at<float>(1,1);
}
/******************************************************
Calculate the vanishing line caused by these four points
******************************************************/
void vanish(Mat P, Mat Q, Mat R, Mat S, Mat& vanish) {
	Mat lrp = Mat (3,1,CV_32F);//line passing through R and P
	Mat lsq = Mat (3,1,CV_32F);//line passing through S and Q
	Mat lqp = Mat (3,1,CV_32F);//line passing through P and Q
	Mat lsr = Mat (3,1,CV_32F);//line passing through R and S

	Mat p1 = Mat (3,1,CV_32F);//intersection of lrp and lsq
	Mat p2 = Mat (3,1,CV_32F);//intersection of lpq and lrs
	cross_product(R,P,lrp);
	cross_product(S,Q,lsq);
	cross_product(Q,P,lqp);
	cross_product(S,R,lsr);
	cross_product(lrp,lsq,p1);
	p1.at<float>(0,0) /= p1.at<float>(2,0);
	p1.at<float>(1,0) /= p1.at<float>(2,0);
	p1.at<float>(2,0) /= p1.at<float>(2,0);
	cross_product(lqp,lsr,p2);
	p2.at<float>(0,0) /= p2.at<float>(2,0);
	p2.at<float>(1,0) /= p2.at<float>(2,0);
	p2.at<float>(2,0) /= p2.at<float>(2,0);
	cross_product(p1,p2,vanish);
}
/*******************************************************************
Calculate cross product between point 1 and point 2
*******************************************************************/
void cross_product(Mat p1, Mat p2, Mat &result) //p1 x p2 = result
{
	float u1 = p1.at<float>(0,0);
	float u2 = p1.at<float>(1,0);
	float u3 = p1.at<float>(2,0);
	float v1 = p2.at<float>(0,0);
	float v2 = p2.at<float>(1,0);
	float v3 = p2.at<float>(2,0);

	result.at<float>(0,0) = u2*v3-v2*u3;
	result.at<float>(1,0) = u3*v1-v3*u1;
	result.at<float>(2,0) = u1*v2-u2*v1;
}
/***********************************
Calculate max and min between two floats
***********************************/
float max(float num1, float num2) {
	return (num1 >= num2)?num1:num2;
}
float min(float num1, float num2) {
	return (num1 <= num2)?num1:num2;
}


/***********************************
Mapping between images by using homography
Map the coordinates from target image into start image, use bilinear interpolation to look for pixel values in start image and fill it into target image
from_img -- the start image
to_img -- the target image
H -- homography matrix
height -- height of the target image
width -- width of the target image
origin_pt -- the orignal point of the target image
LT -- left-top corner coordinate of the start image
BR -- bottom-right corner coordinate of the start image

***********************************/
void Map_homo(Mat H, float height, float width, Point2f origin_pt, Point2f LT, Point2f BR, Mat from_img, Mat &to_img){
    for (float i = origin_pt.y ; i < origin_pt.y +height; i++) {
		for (float j = origin_pt.x ; j < origin_pt.x  + width; j++) {
			
			Mat start_pt = Mat (3,1,CV_32F); //the point in the frame image
			start_pt.at<float>(0,0) = j;
			start_pt.at<float>(1,0) = i;
			start_pt.at<float>(2,0) = 1;
			Mat end_pt = H * start_pt;

			float x_coord = end_pt.at<float>(0,0)/end_pt.at<float>(2,0);
			float y_coord = end_pt.at<float>(1,0)/end_pt.at<float>(2,0);
			//cout <<" LT  " <<LT << " BR = " << BR<<endl;
			if (x_coord < LT.x || x_coord >BR.x || y_coord < LT.y || y_coord > BR.y) {
				/*if (x_coord > 0 && y_coord > 0) {				
					cout <<" X =  " <<x_coord << " Y = " << y_coord<<endl;
				}*/
				continue; //the case that point we want is not inside the face image
			} 
			else {
				bilinear(j,i,x_coord,y_coord,from_img,to_img);
			}		
		}
	}
}

/***************************************
bilinear interpolation fucntion:
--col - column in the dest image
--row - row in the dest image
--x0 - corresponding col location in the face image
--y0 - corresponding row location in the face image
--source - face image
--dest - the image which the face mapped to
****************************************/
void bilinear(float col, float row,float x0,float y0,Mat source, Mat &dest) {
	int x1 = int(x0);
	int x2 = x1 + 1;
	int y1 = int(y0);
	int y2 = y1+1;
	float wx1;
	float wx2;
	float wy1;
	float wy2;
	int height = source.rows;
	int width = source.cols;
	if (x1 < 0 || x1 >= width || x2 < 0 || x2 >= width || y1 < 0 || y1 >= height || y2 < 0 || y2 >= height) { //if go across image boundary
		for (int k = 0; k < 3; k++) {// k = 0, blue channel. k = 1, green channel. k = 2, red channel
			float result = 0;
			dest.at<Vec3b>(row,col)[k] = static_cast<uchar>(result);

		}
	} 
	else {
			wx1 = x0 - x1;
			wx2 = 1 - wx1;
			wy1 = y0 - y1;
			wy2 = 1 - wy1;
			float r1 = wx1 * wy1;
			float r2 = wx2 * wy1;
			float r3 = wx2 * wy2;
			float r4 = wx1 * wy2;

			for (int k = 0; k < 3; k++) {// k = 0, blue channel. k = 1, green channel. k = 2, red channel
	
			uchar a = source.at<Vec3b>(y2,x2)[k];
			uchar b = source.at<Vec3b>(y2,x1)[k];
			uchar c = source.at<Vec3b>(y1,x1)[k];
			uchar d = source.at<Vec3b>(y1,x2)[k];
	
			float result = static_cast<float>(a)*r1+static_cast<float>(b)*r2+static_cast<float>(c)*r3+static_cast<float>(d)*r4;
			if(result >255) {
				result = 255;
			} else if (result < 0) {
				result = 0;
			}
			dest.at<Vec3b>(row,col)[k] = static_cast<uchar>(result);
			}
	}
	
}


int main() {
	Mat flatiron = imread("E:\\2016Fall\\661\\HW3\\flatiron.jpg",1);
	Mat flatiron_proj(flatiron.rows,flatiron.cols, CV_8UC3, Scalar(0,0,0)); //result of removing projective distortion
	Mat flatiron_affine(flatiron.rows,flatiron.cols, CV_8UC3, Scalar(0,0,0));//result of removing affine distortion

	Mat monalisa = imread("E:\\2016Fall\\661\\HW3\\monalisa.jpg",1);
	Mat monalisa_proj(monalisa.rows,monalisa.cols, CV_8UC3, Scalar(0,0,0)); //result of removing projective distortion
	Mat monalisa_affine(monalisa.rows,monalisa.cols, CV_8UC3, Scalar(0,0,0));//result of removing affine distortion

	Mat wideangle = imread("E:\\2016Fall\\661\\HW3\\wideangle.jpg",1);
	Mat wideangle_proj(wideangle.rows,wideangle.cols, CV_8UC3, Scalar(0,0,0)); //result of removing projective distortion
	Mat wideangle_affine(wideangle.rows,wideangle.cols, CV_8UC3, Scalar(0,0,0));//result of affine distortion

	Mat myown = imread("E:\\2016Fall\\661\\HW3\\my_img3.JPG",1);
	Mat myown_proj(myown.rows,myown.cols, CV_8UC3, Scalar(0,0,0)); //result of removing projective distortion
	Mat myown_affine(myown.rows,myown.cols, CV_8UC3, Scalar(0,0,0));//result of removing affine distortion

	Mat myown2 = imread("E:\\2016Fall\\661\\HW3\\IMG_1008.JPG",1);
	Mat myown2_proj(myown2.rows,myown2.cols, CV_8UC3, Scalar(0,0,0)); //result of removing projective distortion
	Mat myown2_affine(myown2.rows,myown2.cols, CV_8UC3, Scalar(0,0,0));//result of removing affine distortion

	float l1;// the bottom row of three parameters to calculate H to remove projective distortion
	float l2;
	float l3;

	Mat l1_x = Mat (3,1,CV_32F); //the first orthogonal pair
	Mat l1_y = Mat (3,1,CV_32F);
	Mat l2_x = Mat (3,1,CV_32F); //the second orthogonal pair
	Mat l2_y = Mat (3,1,CV_32F);
	Mat l3_x = Mat (3,1,CV_32F); //the third orthogonal pair
	Mat l3_y = Mat (3,1,CV_32F);
	Mat l4_x = Mat (3,1,CV_32F); //the fourth orthogonal pair
	Mat l4_y = Mat (3,1,CV_32F);
	Mat l5_x = Mat (3,1,CV_32F); //the fifth orthogonal pair
	Mat l5_y = Mat (3,1,CV_32F);
/**********************************************************************************************
Main function organized as follow:
Task 1: two step method
1. flatiron image
2. monalisa image
3. wideangle image

Task 2: one step method
1. flatiron image
2. monalisa image
3. wideangle image

Task 3: Implement both methods on my own images
1. My first image: two step method
					one step method
2. My second image: two step method
					one step method

Points used in two step method: P,Q,R,S
Additional points used in one step method: A,B,C

index for each image:
1--flatiron
2--monalisa
3--wideangle
4--my first image
5--my second image

For example P_1_3d indicates P point in flatiron

**********************************************************************************************/


/*******************************************************************************************************************************
Task 1. two-steps method on flatiron.jpg, monalisa.jpg, wideangle.jpg
*******************************************************************************************************************************/

/******************************************
For flatiron.jpg image
******************************************/
//Step 1. Remove projective distortion

	//2D format -- (col,row)
	//Four points used to calculate parallel and orthogobal lines
	Point2f P_1_2d = Point2f(112,188); // P -- 0
	Point2f Q_1_2d = Point2f(542,67);// Q -- 1
	Point2f R_1_2d = Point2f(29,492); //  R -- 2
	Point2f S_1_2d = Point2f(578,412);// S -- 3
	//3D format -- (col,row,1)
	Mat P_1_3d = Mat (3,1,CV_32F); // P -- 0
	P_1_3d.at<float>(0,0) = P_1_2d.x;
	P_1_3d.at<float>(1,0) = P_1_2d.y;
	P_1_3d.at<float>(2,0) = 1;
	Mat Q_1_3d = Mat (3,1,CV_32F); // Q -- 1
	Q_1_3d.at<float>(0,0) = Q_1_2d.x;
	Q_1_3d.at<float>(1,0) = Q_1_2d.y;
	Q_1_3d.at<float>(2,0) = 1;
	Mat R_1_3d = Mat (3,1,CV_32F); // R -- 2
	R_1_3d.at<float>(0,0) = R_1_2d.x;
	R_1_3d.at<float>(1,0) = R_1_2d.y;
	R_1_3d.at<float>(2,0) = 1;
	Mat S_1_3d = Mat (3,1,CV_32F); // S -- 3
	S_1_3d.at<float>(0,0) = S_1_2d.x;
	S_1_3d.at<float>(1,0) = S_1_2d.y;
	S_1_3d.at<float>(2,0) = 1;
	
	Mat vanish_1 = Mat (3,1,CV_32F); // vanish line of flatiron.jpg
	Mat H_v_1 = Mat (3,3,CV_32F); //the homography takes vanishing line to infinity
	vanish(P_1_3d,Q_1_3d,R_1_3d,S_1_3d,vanish_1);
	l1 = vanish_1.at<float>(0,0);
	l2 = vanish_1.at<float>(1,0);
	l3 = vanish_1.at<float>(2,0);
	//cout<<vanish_1<<endl<<endl;
	H_v_1.at<float>(0,0) = 1;
	H_v_1.at<float>(1,0) = 0;
	H_v_1.at<float>(2,0) = l1/l3*0.9;
	H_v_1.at<float>(0,1) = 0;
	H_v_1.at<float>(1,1) = 1;
	H_v_1.at<float>(2,1) = l2/l3*0.9;
	H_v_1.at<float>(0,2) = 0;
	H_v_1.at<float>(1,2) = 0;
	H_v_1.at<float>(2,2) = l3/l3*0.9;
	Mat H_v_1_inv = H_v_1.inv();
	Map_homo(H_v_1_inv, float(flatiron.rows), float(flatiron.cols), Point2f(0,0), Point2f(0,0), Point2f(float(flatiron.cols - 1),float(flatiron.rows - 1)), flatiron, flatiron_proj);
	imwrite( "E:\\2016Fall\\661\\HW3\\flatiron_proj.jpg",flatiron_proj);

//Step 2. Remove affine distortion
	//Find two pairs of orthogonal lines
	Mat H_v_1_int_t = H_v_1_inv.t(); //get H^-T to achieve projective distortion removed lines
	Mat RT_1_3d = Mat (3,1,CV_32F); // Right top corner
	RT_1_3d.at<float>(0,0) = 543;
	RT_1_3d.at<float>(1,0) = 67;
	RT_1_3d.at<float>(2,0) = 1;
	Mat LT_1_3d = Mat (3,1,CV_32F); // Left top corner
	LT_1_3d.at<float>(0,0) = 113;
	LT_1_3d.at<float>(1,0) = 188;
	LT_1_3d.at<float>(2,0) = 1;
	Mat RB_1_3d = Mat (3,1,CV_32F); // Right bottom corner
	RB_1_3d.at<float>(0,0) = 579;
	RB_1_3d.at<float>(1,0) = 412;
	RB_1_3d.at<float>(2,0) = 1;
	Mat LB_1_3d = Mat (3,1,CV_32F); // Left bottom corner
	LB_1_3d.at<float>(0,0) = 30;
	LB_1_3d.at<float>(1,0) = 492;
	LB_1_3d.at<float>(2,0) = 1;

	//First pair PQ and PR
	Mat PQ_1 = Mat (3,1,CV_32F);
	Mat PR_1 = Mat (3,1,CV_32F);
    cross_product(RT_1_3d,LT_1_3d,PQ_1);
	cross_product(RT_1_3d,RB_1_3d,PR_1);

	PQ_1 = H_v_1_int_t * PQ_1;//map lines into projective removed image
	PR_1 = H_v_1_int_t * PR_1;

	//Second pair SQ and SR
	Mat SQ_1 = Mat (3,1,CV_32F);
	Mat SR_1 = Mat (3,1,CV_32F);
	cross_product(LB_1_3d,RB_1_3d,SQ_1);
	cross_product(LB_1_3d,LT_1_3d,SR_1);

	SQ_1 = H_v_1_int_t * SQ_1;
	SR_1 = H_v_1_int_t * SR_1;

	//calculate affine homography
	Mat H_a_1 = Mat::eye(3, 3, CV_32F);
	affine_homo(PQ_1,PR_1,SQ_1,SR_1,H_a_1);
	//cout<<H_a_1<<endl<<endl;
	Map_homo(H_v_1_inv * H_a_1, float(flatiron.rows), float(flatiron.cols), Point2f(0,0), Point2f(0,0), Point2f(float(flatiron.cols - 1),float(flatiron.rows - 1)), flatiron,flatiron_affine);
	imwrite( "E:\\2016Fall\\661\\HW3\\flatiron_affine.jpg",flatiron_affine);


	
/******************************************
For monalisa.jpg image
******************************************/
//Step 1. Remove projective distortion

	//2D format -- (col,row)
	//Four points used to calculate parallel and orthogobal lines
	Point2f P_2_2d = Point2f(289,108); // P -- 0
	Point2f Q_2_2d = Point2f(448,144);// Q -- 1
	Point2f R_2_2d = Point2f(254,367); //  R -- 2
	Point2f S_2_2d = Point2f(417,384);// S -- 3
	//3D format -- (col,row,1)
	Mat P_2_3d = Mat (3,1,CV_32F); // P -- 0
	P_2_3d.at<float>(0,0) = P_2_2d.x;
	P_2_3d.at<float>(1,0) = P_2_2d.y;
	P_2_3d.at<float>(2,0) = 1;
	Mat Q_2_3d = Mat (3,1,CV_32F); // Q -- 1
	Q_2_3d.at<float>(0,0) = Q_2_2d.x;
	Q_2_3d.at<float>(1,0) = Q_2_2d.y;
	Q_2_3d.at<float>(2,0) = 1;
	Mat R_2_3d = Mat (3,1,CV_32F); // R -- 2
	R_2_3d.at<float>(0,0) = R_2_2d.x;
	R_2_3d.at<float>(1,0) = R_2_2d.y;
	R_2_3d.at<float>(2,0) = 1;
	Mat S_2_3d = Mat (3,1,CV_32F); // S -- 3
	S_2_3d.at<float>(0,0) = S_2_2d.x;
	S_2_3d.at<float>(1,0) = S_2_2d.y;
	S_2_3d.at<float>(2,0) = 1;
	
	Mat vanish_2 = Mat (3,1,CV_32F); // vanish line of flatiron.jpg
	Mat H_v_2 = Mat (3,3,CV_32F); //the homography takes vanishing line to infinity
	vanish(P_2_3d,Q_2_3d,R_2_3d,S_2_3d,vanish_2);
	l1 = vanish_2.at<float>(0,0);
	l2 = vanish_2.at<float>(1,0);
	l3 = vanish_2.at<float>(2,0);
	H_v_2.at<float>(0,0) = 1;
	H_v_2.at<float>(1,0) = 0;
	H_v_2.at<float>(2,0) = l1/l3*2;
	H_v_2.at<float>(0,1) = 0;
	H_v_2.at<float>(1,1) = 1;
	H_v_2.at<float>(2,1) = l2/l3*2;
	H_v_2.at<float>(0,2) = 0;
	H_v_2.at<float>(1,2) = 0;
	H_v_2.at<float>(2,2) = l3/l3*2;
	Mat H_v_2_inv = H_v_2.inv();
	Map_homo(H_v_2_inv, float(monalisa.rows), float(monalisa.cols), Point2f(0,0), Point2f(0,0), Point2f(float(monalisa.cols - 1),float(monalisa.rows - 1)), monalisa, monalisa_proj);
	imwrite( "E:\\2016Fall\\661\\HW3\\monalisa_proj.jpg",monalisa_proj);
//Step 2. Remove affine distortion

	//Find two pairs of orthogonal lines
	Mat H_v_2_int_t = H_v_2_inv.t(); //get H^-T to achieve projective distortion removed lines
	Mat RT_2_3d = Mat (3,1,CV_32F); // Right top corner
	Q_2_3d.copyTo(RT_2_3d);

	Mat LT_2_3d = Mat (3,1,CV_32F); // Left top corner
	P_2_3d.copyTo(LT_2_3d);

	Mat RB_2_3d = Mat (3,1,CV_32F); // Right bottom corner
	S_2_3d.copyTo(RB_2_3d);

	Mat LB_2_3d = Mat (3,1,CV_32F); // Left bottom corner
	R_2_3d.copyTo(LB_2_3d);

	//First pair PQ and PR
	Mat PQ_2 = Mat (3,1,CV_32F);
	Mat PR_2 = Mat (3,1,CV_32F);
    cross_product(RT_2_3d,LT_2_3d,PQ_2);
	cross_product(RT_2_3d,RB_2_3d,PR_2);

	PQ_2 = H_v_2_int_t * PQ_2;
	PR_2 = H_v_2_int_t * PR_2;
	//Second pair SQ and SR
	Mat SQ_2 = Mat (3,1,CV_32F);
	Mat SR_2 = Mat (3,1,CV_32F);
	cross_product(LB_2_3d,RB_2_3d,SQ_2);
	cross_product(LB_2_3d,LT_2_3d,SR_2);

	SQ_2 = H_v_2_int_t * SQ_2;
	SR_2 = H_v_2_int_t * SR_2;

	//calculate affine homography
	Mat H_a_2 = Mat::eye(3, 3, CV_32F);
	affine_homo(PQ_2,PR_2,SQ_2,SR_2,H_a_2);
	//cout<<H_a_2<<endl<<endl;
	Map_homo(H_v_2_inv * H_a_2, float(monalisa.rows), float(monalisa.cols), Point2f(0,0), Point2f(0,0), Point2f(float(monalisa.cols - 1),float(monalisa.rows - 1)), monalisa,monalisa_affine);
	imwrite( "E:\\2016Fall\\661\\HW3\\monalisa_affine.jpg",monalisa_affine);


/******************************************
For wideangle.jpg image
******************************************/
//Step 1. Remove projective distortion

	//2D format -- (col,row)
	//Four points used to calculate parallel and orthogobal lines
	Point2f P_3_2d = Point2f(47,221); // P -- 0
	Point2f Q_3_2d = Point2f(413,309);// Q -- 1
	Point2f R_3_2d = Point2f(38,297); //  R -- 2
	Point2f S_3_2d = Point2f(405,350);// S -- 3
	//3D format -- (col,row,1)
	Mat P_3_3d = Mat (3,1,CV_32F); // P -- 0
	P_3_3d.at<float>(0,0) = P_3_2d.x;
	P_3_3d.at<float>(1,0) = P_3_2d.y;
	P_3_3d.at<float>(2,0) = 1;
	Mat Q_3_3d = Mat (3,1,CV_32F); // Q -- 1
	Q_3_3d.at<float>(0,0) = Q_3_2d.x;
	Q_3_3d.at<float>(1,0) = Q_3_2d.y;
	Q_3_3d.at<float>(2,0) = 1;
	Mat R_3_3d = Mat (3,1,CV_32F); // R -- 2
	R_3_3d.at<float>(0,0) = R_3_2d.x;
	R_3_3d.at<float>(1,0) = R_3_2d.y;
	R_3_3d.at<float>(2,0) = 1;
	Mat S_3_3d = Mat (3,1,CV_32F); // S -- 3
	S_3_3d.at<float>(0,0) = S_3_2d.x;
	S_3_3d.at<float>(1,0) = S_3_2d.y;
	S_3_3d.at<float>(2,0) = 1;
	
	Mat vanish_3 = Mat (3,1,CV_32F); // vanish line of flatiron.jpg
	Mat H_v_3 = Mat (3,3,CV_32F); //the homography takes vanishing line to infinity
	vanish(P_3_3d,Q_3_3d,R_3_3d,S_3_3d,vanish_3);
	l1 = vanish_3.at<float>(0,0);
	l2 = vanish_3.at<float>(1,0);
	l3 = vanish_3.at<float>(2,0);
	H_v_3.at<float>(0,0) = 1;
	H_v_3.at<float>(1,0) = 0;
	H_v_3.at<float>(2,0) = l1/l3*2;
	H_v_3.at<float>(0,1) = 0;
	H_v_3.at<float>(1,1) = 1;
	H_v_3.at<float>(2,1) = l2/l3*2;
	H_v_3.at<float>(0,2) = 0;
	H_v_3.at<float>(1,2) = 0;
	H_v_3.at<float>(2,2) = l3/l3*2;
	Mat H_v_3_inv = H_v_3.inv();
	Map_homo(H_v_3_inv, float(wideangle.rows), float(wideangle.cols), Point2f(0,0), Point2f(0,0), Point2f(float(wideangle.cols - 1),float(wideangle.rows - 1)), wideangle, wideangle_proj);
	imwrite( "E:\\2016Fall\\661\\HW3\\wideangle_proj.jpg",wideangle_proj);

//Step 2. Remove affine distortion
	//Find two pairs of orthogonal lines
	Mat H_v_3_int_t = H_v_3_inv.t(); //get H^-T to achieve projective distortion removed lines

	Mat RT_3_3d = Mat (3,1,CV_32F); // Right top corner
	Q_3_3d.copyTo(RT_3_3d);
	RT_3_3d = H_v_3 * RT_3_3d;

	Mat LT_3_3d = Mat (3,1,CV_32F); // Left top corner
	P_3_3d.copyTo(LT_3_3d);
	LT_3_3d = H_v_3 * LT_3_3d;

	Mat RB_3_3d = Mat (3,1,CV_32F); // Right bottom corner
	S_3_3d.copyTo(RB_3_3d);
	RB_3_3d = H_v_3 * RB_3_3d;

	Mat LB_3_3d = Mat (3,1,CV_32F); // Left bottom corner
	R_3_3d.copyTo(LB_3_3d);
	LB_3_3d = H_v_3 * LB_3_3d;

	//First pair PQ and PR
	Mat PQ_3 = Mat (3,1,CV_32F);
	Mat PR_3 = Mat (3,1,CV_32F);
    cross_product(RT_3_3d,LT_3_3d,PQ_3);
	cross_product(RT_3_3d,RB_3_3d,PR_3);

	//Second pair SQ and SR
	Mat SQ_3 = Mat (3,1,CV_32F);
	Mat SR_3 = Mat (3,1,CV_32F);
	cross_product(LB_3_3d,RB_3_3d,SQ_3);
	cross_product(LB_3_3d,LT_3_3d,SR_3);

	//calculate affine homography
	Mat H_a_3 = Mat::eye(3, 3, CV_32F);
	affine_homo(PQ_3,PR_3,SQ_3,SR_3,H_a_3);
	//cout<<H_a_3<<endl<<endl;
	Map_homo(H_v_3_inv * H_a_3, float(wideangle.rows), float(wideangle.cols), Point2f(0,0), Point2f(0,0), Point2f(float(wideangle.cols - 1),float(wideangle.rows - 1)), wideangle,wideangle_affine);
	imwrite( "E:\\2016Fall\\661\\HW3\\wideangle_affine.jpg",wideangle_affine);



/******************************************************************************************************************************
Task 2 one step method
*******************************************************************************************************************************/
//**********************************************************
//For flatiron 
	Mat flatiron_one(flatiron.rows,flatiron.cols, CV_8UC3, Scalar(0,0,0));
	Point2f A_1_2d = Point2f(164,384); // Used 3 more points for orthognal pairs construction
	Point2f B_1_2d = Point2f(146,467);// 
	Point2f C_1_2d = Point2f(233,456); //  
	//3D format -- (col,row,1)
	Mat A_1_3d = Mat (3,1,CV_32F); 
	A_1_3d.at<float>(0,0) = A_1_2d.x;
	A_1_3d.at<float>(1,0) = A_1_2d.y;
	A_1_3d.at<float>(2,0) = 1;
	Mat B_1_3d = Mat (3,1,CV_32F); 
	B_1_3d.at<float>(0,0) = B_1_2d.x;
	B_1_3d.at<float>(1,0) = B_1_2d.y;
	B_1_3d.at<float>(2,0) = 1;
	Mat C_1_3d = Mat (3,1,CV_32F); 
	C_1_3d.at<float>(0,0) = C_1_2d.x;
	C_1_3d.at<float>(1,0) = C_1_2d.y;
	C_1_3d.at<float>(2,0) = 1;

	//calculate each line from two points
	cross_product(R_1_3d,P_1_3d,l1_x);
	cross_product(P_1_3d,Q_1_3d,l1_y);
	cross_product(S_1_3d,Q_1_3d,l2_x);
	cross_product(P_1_3d,Q_1_3d,l2_y);
	cross_product(R_1_3d,P_1_3d,l3_x);
	cross_product(R_1_3d,S_1_3d,l3_y);
	cross_product(S_1_3d,Q_1_3d,l4_x);
	cross_product(R_1_3d,S_1_3d,l4_y);
	cross_product(B_1_3d,A_1_3d,l5_x);
	cross_product(B_1_3d,C_1_3d,l5_y);
	Mat H_U_1 = Mat::eye(3, 3, CV_32F);
	onestep_homo(l1_x,l1_y,l2_x,l2_y,l3_x,l3_y,l4_x,l4_y,l5_x, l5_y,H_U_1);
	H_U_1.at<float>(2,2) = 1;
	Map_homo(H_U_1, float(flatiron.rows), float(flatiron.cols), Point2f(0,0), Point2f(0,0), Point2f(float(flatiron.cols - 1),float(flatiron.rows - 1)), flatiron, flatiron_one);
	imwrite("E:\\2016Fall\\661\\HW3\\flatiron_one.jpg",flatiron_one);

//*******************************************************************
//For monalisa.jpg 
	Mat monalisa_one(monalisa.rows,monalisa.cols, CV_8UC3, Scalar(0,0,0));
	Point2f A_2_2d = Point2f(171,529); // Used 3 more points for orthognal pairs construction
	Point2f B_2_2d = Point2f(162,599);// 
	Point2f C_2_2d = Point2f(716,597); //  
	//3D format -- (col,row,1)
	Mat A_2_3d = Mat (3,1,CV_32F); 
	A_2_3d.at<float>(0,0) = A_2_2d.x;
	A_2_3d.at<float>(1,0) = A_2_2d.y;
	A_2_3d.at<float>(2,0) = 1;
	Mat B_2_3d = Mat (3,1,CV_32F); 
	B_2_3d.at<float>(0,0) = B_2_2d.x;
	B_2_3d.at<float>(1,0) = B_2_2d.y;
	B_2_3d.at<float>(2,0) = 1;
	Mat C_2_3d = Mat (3,1,CV_32F); 
	C_2_3d.at<float>(0,0) = C_2_2d.x;
	C_2_3d.at<float>(1,0) = C_2_2d.y;
	C_2_3d.at<float>(2,0) = 1;

	//calculate each line from two points
	cross_product(R_2_3d,P_2_3d,l1_x);
	cross_product(P_2_3d,Q_2_3d,l1_y);
	cross_product(S_2_3d,Q_2_3d,l2_x);
	cross_product(P_2_3d,Q_2_3d,l2_y);
	cross_product(R_2_3d,P_2_3d,l3_x);
	cross_product(R_2_3d,S_2_3d,l3_y);
	cross_product(S_2_3d,Q_2_3d,l4_x);
	cross_product(R_2_3d,S_2_3d,l4_y);
	cross_product(B_2_3d,A_2_3d,l5_x);
	cross_product(B_2_3d,C_2_3d,l5_y);
	Mat H_U_2 = Mat::eye(3, 3, CV_32F);
	onestep_homo(l1_x,l1_y,l2_x,l2_y,l3_x,l3_y,l4_x,l4_y,l5_x, l5_y,H_U_2);

	H_U_2.at<float>(2,2) = 0.3;
	H_U_2.at<float>(0,2) -= 200;
	//H_U_2.at<float>(1,2) += 200;
	Map_homo(H_U_2, float(monalisa.rows), float(monalisa.cols), Point2f(0,0), Point2f(0,0), Point2f(float(monalisa.cols - 1),float(monalisa.rows - 1)), monalisa, monalisa_one);
	imwrite("E:\\2016Fall\\661\\HW3\\monalisa_one.jpg",monalisa_one);

//*******************************************************************
//For wideangle.jpg 
	Mat wideangle_one(wideangle.rows,wideangle.cols, CV_8UC3, Scalar(0,0,0));
	Point2f A_3_2d = Point2f(292,287); // Used 3 more points for orthognal pairs construction
	Point2f B_3_2d = Point2f(283,346);// 
	Point2f C_3_2d = Point2f(328,349); //  
	//3D format -- (col,row,1)
	Mat A_3_3d = Mat (3,1,CV_32F); 
	A_3_3d.at<float>(0,0) = A_3_2d.x;
	A_3_3d.at<float>(1,0) = A_3_2d.y;
	A_3_3d.at<float>(2,0) = 1;
	Mat B_3_3d = Mat (3,1,CV_32F); 
	B_3_3d.at<float>(0,0) = B_3_2d.x;
	B_3_3d.at<float>(1,0) = B_3_2d.y;
	B_3_3d.at<float>(2,0) = 1;
	Mat C_3_3d = Mat (3,1,CV_32F); 
	C_3_3d.at<float>(0,0) = C_3_2d.x;
	C_3_3d.at<float>(1,0) = C_3_2d.y;
	C_3_3d.at<float>(2,0) = 1;

	cross_product(R_3_3d,P_3_3d,l1_x);

	cross_product(P_3_3d,Q_3_3d,l1_y);
	cross_product(S_3_3d,Q_3_3d,l2_x);
	cross_product(P_3_3d,Q_3_3d,l2_y);
	cross_product(R_3_3d,P_3_3d,l3_x);
	cross_product(R_3_3d,S_3_3d,l3_y);
	cross_product(S_3_3d,Q_3_3d,l4_x);
	cross_product(R_3_3d,S_3_3d,l4_y);
	cross_product(B_3_3d,A_3_3d,l5_x);
	cross_product(B_3_3d,C_3_3d,l5_y);
	Mat H_U_3 = Mat::eye(3, 3, CV_32F);
	onestep_homo(l1_x,l1_y,l2_x,l2_y,l3_x,l3_y,l4_x,l4_y,l5_x,l5_y,H_U_3);

	Map_homo(H_U_3, float(wideangle.rows), float(wideangle.cols), Point2f(0,0), Point2f(0,0), Point2f(float(wideangle.cols - 1),float(wideangle.rows - 1)), wideangle, wideangle_one);
	imwrite("E:\\2016Fall\\661\\HW3\\wideangle_one.jpg",wideangle_one);



/************************************************************************************************************************
Task 3. Use my own image to implement two-step and one-step methods
************************************************************************************************************************/

/******************************************
First image of my own
******************************************/

//Two step method-------------------------------------------------------------------------------
//Step 1. Remove projective distortion

	//2D format -- (col,row)

	Point2f P_4_2d = Point2f(1090,314); // P -- 0
	Point2f Q_4_2d = Point2f(2190,405);// Q -- 1
	Point2f R_4_2d = Point2f(1091,1173); //  R -- 2
	Point2f S_4_2d = Point2f(2293,1199);// S -- 3
	//3D format -- (col,row,1)
	Mat P_4_3d = Mat (3,1,CV_32F); // P -- 0
	P_4_3d.at<float>(0,0) = P_4_2d.x;
	P_4_3d.at<float>(1,0) = P_4_2d.y;
	P_4_3d.at<float>(2,0) = 1;
	Mat Q_4_3d = Mat (3,1,CV_32F); // Q -- 1
	Q_4_3d.at<float>(0,0) = Q_4_2d.x;
	Q_4_3d.at<float>(1,0) = Q_4_2d.y;
	Q_4_3d.at<float>(2,0) = 1;
	Mat R_4_3d = Mat (3,1,CV_32F); // R -- 2
	R_4_3d.at<float>(0,0) = R_4_2d.x;
	R_4_3d.at<float>(1,0) = R_4_2d.y;
	R_4_3d.at<float>(2,0) = 1;
	Mat S_4_3d = Mat (3,1,CV_32F); // S -- 3
	S_4_3d.at<float>(0,0) = S_4_2d.x;
	S_4_3d.at<float>(1,0) = S_4_2d.y;
	S_4_3d.at<float>(2,0) = 1;
	
	Mat vanish_4 = Mat (3,1,CV_32F); // vanish line of flatiron.jpg
	Mat H_v_4 = Mat (3,3,CV_32F); //the homography takes vanishing line to infinity
	vanish(P_4_3d,Q_4_3d,R_4_3d,S_4_3d,vanish_4);
	l1 = vanish_4.at<float>(0,0);
	l2 = vanish_4.at<float>(1,0);
	l3 = vanish_4.at<float>(2,0);
	H_v_4.at<float>(0,0) = 1;
	H_v_4.at<float>(1,0) = 0;
	H_v_4.at<float>(2,0) = l1/l3*2;
	H_v_4.at<float>(0,1) = 0;
	H_v_4.at<float>(1,1) = 1;
	H_v_4.at<float>(2,1) = l2/l3*2;
	H_v_4.at<float>(0,2) = 0;
	H_v_4.at<float>(1,2) = 0;
	H_v_4.at<float>(2,2) = l3/l3*2;
	//cout<<"This is H3P "<<endl<<H_v_3<<endl<<endl;
	Mat H_v_4_inv = H_v_4.inv();
	//cout<<H_v_1_inv<<endl<<endl;
	Map_homo(H_v_4_inv, float(myown.rows), float(myown.cols), Point2f(0,0), Point2f(0,0), Point2f(float(myown.cols - 1),float(myown.rows - 1)), myown, myown_proj);
	imwrite( "E:\\2016Fall\\661\\HW3\\myown_proj.jpg",myown_proj);

//Step 2. Remove affine distortion
	//Find two pairs of orthogonal lines
	Mat H_v_4_int_t = H_v_4_inv.t(); //get H^-T to achieve projective distortion removed lines

	Mat RT_4_3d = Mat (3,1,CV_32F); // Right top corner
	Q_4_3d.copyTo(RT_4_3d);
	RT_4_3d = H_v_4 * RT_4_3d;

	Mat LT_4_3d = Mat (3,1,CV_32F); // Left top corner
	P_4_3d.copyTo(LT_4_3d);
	LT_4_3d = H_v_4 * LT_4_3d;

	Mat RB_4_3d = Mat (3,1,CV_32F); // Right bottom corner
	S_4_3d.copyTo(RB_4_3d);
	RB_4_3d = H_v_4 * RB_4_3d;

	Mat LB_4_3d = Mat (3,1,CV_32F); // Left bottom corner
	R_4_3d.copyTo(LB_4_3d);
	LB_4_3d = H_v_4 * LB_4_3d;

	//First pair PQ and PR
	Mat PQ_4 = Mat (3,1,CV_32F);
	Mat PR_4 = Mat (3,1,CV_32F);
    cross_product(RT_4_3d,LT_4_3d,PQ_4);
	cross_product(RT_4_3d,RB_4_3d,PR_4);

	//Second pair SQ and SR
	Mat SQ_4 = Mat (3,1,CV_32F);
	Mat SR_4 = Mat (3,1,CV_32F);
	cross_product(LB_4_3d,RB_4_3d,SQ_4);
	cross_product(LB_4_3d,LT_4_3d,SR_4);


	//calculate affine homography
	Mat H_a_4 = Mat::eye(3, 3, CV_32F);
	affine_homo(PQ_4,PR_4,SQ_4,SR_4,H_a_4);
	//cout<<H_a_3<<endl<<endl;
	Map_homo(H_v_4_inv * H_a_4, float(myown.rows), float(myown.cols), Point2f(0,0), Point2f(0,0), Point2f(float(myown.cols - 1),float(myown.rows - 1)), myown,myown_affine);
	imwrite( "E:\\2016Fall\\661\\HW3\\myown_affine.jpg",myown_affine);




//One step method----------------------------------------------------------------------------------------
	Mat myown_one(myown.rows,myown.cols, CV_8UC3, Scalar(0,0,0));
	Point2f A_4_2d = Point2f(1342,1394); // Used 3 more points for orthognal pairs construction
	Point2f B_4_2d = Point2f(1362,2043);// 
	Point2f C_4_2d = Point2f(1952,2011); //  
	//3D format -- (col,row,1)
	Mat A_4_3d = Mat (3,1,CV_32F); 
	A_4_3d.at<float>(0,0) = A_4_2d.x;
	A_4_3d.at<float>(1,0) = A_4_2d.y;
	A_4_3d.at<float>(2,0) = 1;
	Mat B_4_3d = Mat (3,1,CV_32F); 
	B_4_3d.at<float>(0,0) = B_4_2d.x;
	B_4_3d.at<float>(1,0) = B_4_2d.y;
	B_4_3d.at<float>(2,0) = 1;
	Mat C_4_3d = Mat (3,1,CV_32F); 
	C_4_3d.at<float>(0,0) = C_4_2d.x;
	C_4_3d.at<float>(1,0) = C_4_2d.y;
	C_4_3d.at<float>(2,0) = 1;

	//calculate each line from two points
	cross_product(R_4_3d,P_4_3d,l1_x);
	cross_product(P_4_3d,Q_4_3d,l1_y);
	cross_product(S_4_3d,Q_4_3d,l2_x);
	cross_product(P_4_3d,Q_4_3d,l2_y);
	cross_product(R_4_3d,P_4_3d,l3_x);
	cross_product(R_4_3d,S_4_3d,l3_y);
	cross_product(S_4_3d,Q_4_3d,l4_x);
	cross_product(R_4_3d,S_4_3d,l4_y);
	cross_product(B_4_3d,A_4_3d,l5_x);
	cross_product(B_4_3d,C_4_3d,l5_y);
	Mat H_U_4 = Mat::eye(3, 3, CV_32F);
	onestep_homo(l1_x,l1_y,l2_x,l2_y,l3_x,l3_y,l4_x,l4_y,l5_x, l5_y,H_U_4);
	H_U_4.at<float>(2,2) = 1;
	Map_homo(H_U_4, float(myown.rows), float(myown.cols), Point2f(0,0), Point2f(0,0), Point2f(float(myown.cols - 1),float(myown.rows - 1)), myown, myown_one);
	imwrite("E:\\2016Fall\\661\\HW3\\myown_one.jpg",myown_one);





/********************************************************************************************************************
For second of my own image
********************************************************************************************************************/
//Two-step method--------------------------------------------------------------------
//Step 1. Remove projective distortion

	//2D format -- (col,row)
	Point2f P_5_2d = Point2f(636,379); // P -- 0
	Point2f Q_5_2d = Point2f(2790,553);// Q -- 1
	Point2f R_5_2d = Point2f(710,939); //  R -- 2
	Point2f S_5_2d = Point2f(2546,1030);// S -- 3
	//3D format -- (col,row,1)
	Mat P_5_3d = Mat (3,1,CV_32F); // P -- 0
	P_5_3d.at<float>(0,0) = P_5_2d.x;
	P_5_3d.at<float>(1,0) = P_5_2d.y;
	P_5_3d.at<float>(2,0) = 1;
	Mat Q_5_3d = Mat (3,1,CV_32F); // Q -- 1
	Q_5_3d.at<float>(0,0) = Q_5_2d.x;
	Q_5_3d.at<float>(1,0) = Q_5_2d.y;
	Q_5_3d.at<float>(2,0) = 1;
	Mat R_5_3d = Mat (3,1,CV_32F); // R -- 2
	R_5_3d.at<float>(0,0) = R_5_2d.x;
	R_5_3d.at<float>(1,0) = R_5_2d.y;
	R_5_3d.at<float>(2,0) = 1;
	Mat S_5_3d = Mat (3,1,CV_32F); // S -- 3
	S_5_3d.at<float>(0,0) = S_5_2d.x;
	S_5_3d.at<float>(1,0) = S_5_2d.y;
	S_5_3d.at<float>(2,0) = 1;
	
	Mat vanish_5 = Mat (3,1,CV_32F); // vanish line of flatiron.jpg
	Mat H_v_5 = Mat (3,3,CV_32F); //the homography takes vanishing line to infinity
	vanish(P_5_3d,Q_5_3d,R_5_3d,S_5_3d,vanish_5);
	l1 = vanish_5.at<float>(0,0);
	l2 = vanish_5.at<float>(1,0);
	l3 = vanish_5.at<float>(2,0);
	H_v_5.at<float>(0,0) = 1;
	H_v_5.at<float>(1,0) = 0;
	H_v_5.at<float>(2,0) = l1/l3*2;
	H_v_5.at<float>(0,1) = 0;
	H_v_5.at<float>(1,1) = 1;
	H_v_5.at<float>(2,1) = l2/l3*2;
	H_v_5.at<float>(0,2) = 0;
	H_v_5.at<float>(1,2) = 0;
	H_v_5.at<float>(2,2) = l3/l3*2;
	//cout<<"This is H3P "<<endl<<H_v_3<<endl<<endl;
	Mat H_v_5_inv = H_v_5.inv();
	//cout<<H_v_1_inv<<endl<<endl;
	Map_homo(H_v_5_inv, float(myown2.rows), float(myown2.cols), Point2f(0,0), Point2f(0,0), Point2f(float(myown2.cols - 1),float(myown2.rows - 1)), myown2, myown2_proj);
	imwrite( "E:\\2016Fall\\661\\HW3\\myown2_proj.jpg",myown2_proj);

//Step 2. Remove affine distortion

	//Find two pairs of orthogonal lines
	Mat H_v_5_int_t = H_v_5_inv.t(); //get H^-T to achieve projective distortion removed lines
	//Four points used to calculate orthogonal lines, first map points from original image into projective distortion removed image
	Mat RT_5_3d = Mat (3,1,CV_32F); // Right top corner
	Q_5_3d.copyTo(RT_5_3d);
	RT_5_3d = H_v_5 * RT_5_3d;//

	Mat LT_5_3d = Mat (3,1,CV_32F); // Left top corner
	P_5_3d.copyTo(LT_5_3d);
	LT_5_3d = H_v_5 * LT_5_3d;

	Mat RB_5_3d = Mat (3,1,CV_32F); // Right bottom corner
	S_5_3d.copyTo(RB_5_3d);
	RB_5_3d = H_v_5 * RB_5_3d;

	Mat LB_5_3d = Mat (3,1,CV_32F); // Left bottom corner
	R_5_3d.copyTo(LB_5_3d);
	LB_5_3d = H_v_5 * LB_5_3d;

	//first orthogonal pairs PQ and PR
	Mat PQ_5 = Mat (3,1,CV_32F);
	Mat PR_5 = Mat (3,1,CV_32F);
    cross_product(RT_5_3d,LT_5_3d,PQ_5);
	cross_product(RT_5_3d,RB_5_3d,PR_5);

	//Second pair SQ and SR
	Mat SQ_5 = Mat (3,1,CV_32F);
	Mat SR_5 = Mat (3,1,CV_32F);
	cross_product(LB_5_3d,RB_5_3d,SQ_5);
	cross_product(LB_5_3d,LT_5_3d,SR_5);


	//calculate affine homography
	Mat H_a_5 = Mat::eye(3, 3, CV_32F);
	affine_homo(PQ_5,PR_5,SQ_5,SR_5,H_a_5);
	//cout<<H_a_3<<endl<<endl;
	Map_homo(H_v_5_inv * H_a_5, float(myown2.rows), float(myown2.cols), Point2f(0,0), Point2f(0,0), Point2f(float(myown2.cols - 1),float(myown2.rows - 1)), myown2,myown2_affine);
	imwrite( "E:\\2016Fall\\661\\HW3\\myown2_affine.jpg",myown2_affine);


//One-step method
	Mat myown2_one(myown2.rows,myown2.cols, CV_8UC3, Scalar(0,0,0));
	Point2f A_5_2d = Point2f(721,1008); // Used 3 more points for orthognal pairs construction
	Point2f B_5_2d = Point2f(805,1673);// 
	Point2f C_5_2d = Point2f(2216,1693); //  
	//3D format -- (col,row,1)
	Mat A_5_3d = Mat (3,1,CV_32F); 
	A_5_3d.at<float>(0,0) = A_5_2d.x;
	A_5_3d.at<float>(1,0) = A_5_2d.y;
	A_5_3d.at<float>(2,0) = 1;
	Mat B_5_3d = Mat (3,1,CV_32F); 
	B_5_3d.at<float>(0,0) = B_5_2d.x;
	B_5_3d.at<float>(1,0) = B_5_2d.y;
	B_5_3d.at<float>(2,0) = 1;
	Mat C_5_3d = Mat (3,1,CV_32F); 
	C_5_3d.at<float>(0,0) = C_5_2d.x;
	C_5_3d.at<float>(1,0) = C_5_2d.y;
	C_5_3d.at<float>(2,0) = 1;

	//calculate each line from two points, totally 5 pairs of orthogonal lines
	cross_product(R_5_3d,P_5_3d,l1_x);
	cross_product(P_5_3d,Q_5_3d,l1_y);
	cross_product(S_5_3d,Q_5_3d,l2_x);
	cross_product(P_5_3d,Q_5_3d,l2_y);
	cross_product(R_5_3d,P_5_3d,l3_x);
	cross_product(R_5_3d,S_5_3d,l3_y);
	cross_product(S_5_3d,Q_5_3d,l4_x);
	cross_product(R_5_3d,S_5_3d,l4_y);
	cross_product(B_5_3d,A_5_3d,l5_x);
	cross_product(B_5_3d,C_5_3d,l5_y);
	Mat H_U_5 = Mat::eye(3, 3, CV_32F);
	onestep_homo(l1_x,l1_y,l2_x,l2_y,l3_x,l3_y,l4_x,l4_y,l5_x, l5_y,H_U_5);//calculate the homography
	Map_homo(H_U_5, float(myown2.rows), float(myown2.cols), Point2f(0,0), Point2f(0,0), Point2f(float(myown2.cols - 1),float(myown2.rows - 1)), myown2, myown2_one);
	imwrite("E:\\2016Fall\\661\\HW3\\myown2_one.jpg",myown2_one);


  	return 1;
}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      