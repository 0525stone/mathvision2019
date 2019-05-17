#include "polygon_demo.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
// for generating random number
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>

using namespace std;
using namespace cv;

PolygonDemo::PolygonDemo()
{
	m_data_ready = false;
}

PolygonDemo::~PolygonDemo()
{
}

void PolygonDemo::refreshWindow()
{
	Mat frame = Mat::zeros(480, 640, CV_8UC3);
	if (!m_data_ready)
		putText(frame, "Input data points (double click: finish)", Point(10, 470), FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 148, 0), 1);

	drawPolygon(frame, m_data_pts, m_data_ready);
	if (m_data_ready)
	{
		// polygon area
		// 넓이 출력해주는 파트
		if (m_param.compute_area)
		{
			int area = polyArea(m_data_pts);
			char str[100];
			sprintf_s(str, 100, "Area = %d", area);
			putText(frame, str, Point(25, 25), FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
			printf("출력 완료\n");
		}

		// pt in polygon
		if (m_param.check_ptInPoly)
		{
			for (int i = 0; i < (int)m_test_pts.size(); i++)
			{
				if (ptInPolygon(m_data_pts, m_test_pts[i]))
				{
					circle(frame, m_test_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
				}
				else
				{
					circle(frame, m_test_pts[i], 2, Scalar(128, 128, 128), CV_FILLED);
				}
			}
		}

		// homography check
		if (m_param.check_homography && m_data_pts.size() == 4)
		{
			// rect points
			int rect_sz = 100;
			vector<Point> rc_pts;
			rc_pts.push_back(Point(0, 0));
			rc_pts.push_back(Point(0, rect_sz));
			rc_pts.push_back(Point(rect_sz, rect_sz));
			rc_pts.push_back(Point(rect_sz, 0));
			rectangle(frame, Rect(0, 0, rect_sz, rect_sz), Scalar(255, 255, 255), 1);

			// draw mapping
			char* abcd[4] = { "A", "B", "C", "D" };
			for (int i = 0; i < 4; i++)
			{
				line(frame, rc_pts[i], m_data_pts[i], Scalar(255, 0, 0), 1);
				circle(frame, rc_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
				circle(frame, m_data_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
				putText(frame, abcd[i], m_data_pts[i], FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
			}

			// check homography  classifyHomography가 뭐길래 이거의 출력(homo_type)이 바로 형태를 정의해주지..
			int homo_type = classifyHomography(rc_pts, m_data_pts);
			char type_str[100];
			switch (homo_type)
			{
			case NORMAL:
				sprintf_s(type_str, 100, "normal");
				break;
			case CONCAVE:
				sprintf_s(type_str, 100, "concave");
				break;
			case TWIST:
				sprintf_s(type_str, 100, "twist");
				break;
			case REFLECTION:
				sprintf_s(type_str, 100, "reflection");
				break;
			case CONCAVE_REFLECTION:
				sprintf_s(type_str, 100, "concave reflection");
				break;
			}

			putText(frame, type_str, Point(15, 125), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
		}

		// fit line
		if (m_param.fit_line)
		{
			Point2d point_dep, point_arr;
			Point2d point_depa, point_arra;
			Point2d p_in_depa, p_in_arra;
			int n_itr = 10;

			// parameter for RANSAC
			//int ransac_itr = 100;
			double a, b, c,th;


			std::vector<cv::Point2d> pointrb_dep, pointrb_arr;
		//	bool ok_axb = fitLine_axb(m_data_pts, point_dep, point_arr);
			bool ok_axby = fitLine_axby(m_data_pts, point_depa, point_arra);
			bool ok_axb = false; //  , ok_robust = false
		//	bool ok_axby = false; 
		//	bool ok_robust = fitLine_robust(m_data_pts, pointrb_dep, pointrb_arr, n_itr);
			bool ok_robust = false;
			bool ok_RANSAC = fitLine_RANSAC(m_data_pts, a, b, c,th);

			if (ok_axb){
				line(frame, point_dep, point_arr, Scalar(0, 255, 0), 1);
				putText(frame, "y=ax+b", Point(5, 25), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1);

			}
			if (ok_axby){
				line(frame, point_depa, point_arra, Scalar(0, 0, 255), 1);
				putText(frame, "ax+by+c = 0", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 1);
			}
			if (ok_robust){
				for (int n = 0; n < n_itr; n++)
				{
					// 색깔 변수에도 변화를 줌으로써 분간 되게끔
					string text_ = "iteration : "+std::to_string(n+1);
					line(frame, pointrb_dep[n], pointrb_arr[n], Scalar(0, 255 / n_itr*n, 255), 1);
					putText(frame, text_, Point(5, 45+n*20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255/n_itr*n, 255), 1);
				}	
			}

			std::vector<cv::Point> p_inlier;
			double d_ = sqrt(pow(a, 2) + pow(b, 2));
			printf("a : %f, b : %f, c: %f\n",a,b,c);

			if (ok_RANSAC){
				for (int j = 0; j < m_data_pts.size(); j++){
					double res = abs((a*m_data_pts[j].x + b*m_data_pts[j].y + c) / d_);
					cout << res << endl;
					if (res < th){
						// circle 들어가야하는 부분
						circle(frame, m_data_pts[j], 5, Scalar(255, 0, 0), CV_FILLED);
						p_inlier.push_back(cv::Point(m_data_pts[j]));
					}
				}
				line(frame, Point(0, (-a * 0 - c) / b), Point(640, (-a * 640 - c) / b), Scalar(0, 255, 0), 1);
				putText(frame, "RANSAC", Point(5, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1);
				bool try_axby = fitLine_axby(p_inlier, p_in_depa, p_in_arra);
				if (try_axby){
					line(frame, p_in_depa, p_in_arra, Scalar(255, 0, 0), 1);
					putText(frame, "LS with inliers", Point(5, 85), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 1);
				}

			}
		}

		// fit circle
		if (m_param.fit_circle)
		{
			Point2d center;
			double radius = 0;
			bool ok = fitCircle(m_data_pts, center, radius);
			if (ok)
			{
				circle(frame, center, (int)(radius + 0.5), Scalar(0, 255, 0), 1);
				circle(frame, center, 2, Scalar(0, 255, 0), CV_FILLED);
			}
		}
		// fit Ellipse
		if (m_param.fit_ellipse)
		{
			// 여기도 다 구현.
			// 타원을 그리기 위해서 적어도 필요한건...? - 5개의 계수 쌍곡선의 방정식으로 푸는 거니까
			Point2d center_e;
			Point2d v;	// v는 사이즈 위한 변수
			double angle = 0, startAngle = 0, endAngle = 360;
			int  arcStart = 0, arcEnd = 0, delta = 0;
			RotatedRect rec_e; // center , size, angle 이 출력됨 (ellipse에 필요한 모든것)
			// 예시 : ellipse(frame,center,Size axes, double angle, double startAngle, double endAngle,color, thickness,linetype,shift)
			bool ok = fitEllipse(m_data_pts, center_e, v);
			std::vector<cv::Point> points_e;
			float axes_scale = 0.5;
			//	bool ok = true;
			if (ok)
			{
				//printf("center of ellipse :(%f %f)\naxes: %f %f\n", center_e.x, center_e.y, v.x, v.y);
				rec_e = cv::fitEllipse(m_data_pts);  // rectangle 을 출력해준대
				//ellipse2Poly(rec_e.center, rec_e.size, rec_e.angle, arcStart, arcEnd, delta, points_e); //vector<Point>& pts
				printf("done\n");
				cv::Size axes(rec_e.size.width*axes_scale, rec_e.size.height * axes_scale);
				ellipse(frame, rec_e.center, axes, rec_e.angle, startAngle, endAngle, Scalar(0, 255, 255), 1);
			}
		}
	}

	imshow("PolygonDemo", frame);
}
bool PolygonDemo::fitLine_RANSAC(const std::vector<cv::Point>& pts, double& a, double& b, double& c, double& th)
{
	// 2019.05.16 RANSAC code	-  a,b,c가 결과로 나와서 넘어가게끔
	// parameter for RANSAC
	int m = 2, cnt = 0, N_itr; // m:sampling 갯수, cnt : inlier 갯수, N : iteration
	int cnt_max = 0; //cnt_max : inlier 최대
	double alpha = 0.8, res,T; // alpha : inlier 비율, res : residual, T : threshold for residual 
	// parameter for line fitting
	cv::Mat A(m, 3, CV_64F);
	cv::Mat w, u, v_t, sv;
	int n = (int)pts.size();
	if (n < m) return false;

	T = 50; // T 정의
	double p = 1 / (alpha*n) * 1 / (alpha*n - 1);
	double p_in = 1 - pow((1 - p), m);
	N_itr = (int)1/p_in; // iteration 계산
	//N_itr = 1000; 
	cout << N_itr<<endl;
	for (int i = 0; i < N_itr; i++){           // RANSAC iteration 파트
		// 이 안에 ax+by+c=0 fitting파트, inlier 갯수 세는 파트
	//	printf("get in and start\n");
	// random sampling
		for (int r = 0; r < m; r++){ // 중복이 많이 발생함
			int idx = rand() % n;
		//	cout << idx << endl;
			A.at<double>(r, 0) = pts[idx].x;
			A.at<double>(r, 1) = pts[idx].y;
			A.at<double>(r, 2) = 1;
		}
	// model estimation
		// PCA의 right singular vector구해야함
		SVD::compute(A, w, u, v_t, SVD::FULL_UV);
		sv = v_t.row(2);		// v_t의 마지막 행
		double a_temp = sv.at<double>(0, 0);
		double b_temp = sv.at<double>(0, 1);
		double c_temp = sv.at<double>(0, 2);
		double d_ = sqrt(pow(a_temp, 2) + pow(b_temp, 2) );
	// counting inlier
		cnt = 0;
		for (int j = 0; j < n; j++){
			res = abs((a_temp*pts[j].x + b_temp*pts[j].y + c_temp)/d_);
			if (res < T){ 
				cnt++;
			//	cout << res << endl;
			}
		}
		if (cnt>cnt_max){
			cnt_max = cnt;
			a = a_temp;
			b = b_temp;
			c = c_temp;
		}
	}
	th = T;
	printf("cmax = %d\n", cnt_max);
	return true;
}


bool PolygonDemo::fitLine_robust(const std::vector<cv::Point>& pts, std::vector<cv::Point2d>& pointrb_dep, std::vector<cv::Point2d>& pointrb_arr, int& itr)
{
	// 2019.05.03 line fitting with robust method
	// pointrb_dep,arr 둘다 vector 형이니까 iteration 몇번 한 점들까지 보내주기
	int n = (int)pts.size();
	Mat p, A_inv, r, w_cau, AtWA, AtWA_inv;
	Mat A = Mat::ones(n, 2, CV_64F);
	Mat Y = Mat::zeros(n, 1, CV_64F);
	Mat W = Mat::zeros(n, n, CV_64F);
	
	// 입력 data 로 A, y 
	for (int i = 0; i < n; i++) {
		int x = pts[i].x;
		int y = pts[i].y;
		A.at<double>(i, 0) = x;
		Y.at<double>(i, 0) = y;
	}

	// initial value (residual 계산 불가하므로)

	invert(A, A_inv, DECOMP_SVD);
	p = A_inv * Y;
	double a = p.at<double>(0, 0); // p에 a,b들 들어감
	double b = p.at<double>(1, 0);
	pointrb_dep.push_back(cv::Point2d(0, a * 0 + b));
	pointrb_arr.push_back(cv::Point2d(640, a * 640 + b));

	for (int i = 1; i < itr; i++){
		r = A * p - Y; // residual
		w_cau = 1 / (abs(r) / 1.3998 + 1); // cauchu weight function
		for (int i = 0; i < n; i++) {
			W.at<double>(i, i) = w_cau.at<double>(i, 0); // W diagonal matrix with w_cau
		}
		AtWA = A.t() * W * A;
		invert(AtWA, AtWA_inv, DECOMP_SVD);
		p = AtWA_inv * A.t() * W * Y;
		a = p.at<double>(0, 0); // p에 a,b들 들어감
		b = p.at<double>(1, 0);
		pointrb_dep.push_back(cv::Point2d(0, a * 0 + b));
		pointrb_arr.push_back(cv::Point2d(640, a * 640 + b));
	}
	return true;
}

// return the area of polygon
int PolygonDemo::polyArea(const std::vector<cv::Point>& vtx)
{
	// 여기가 넓이 구해주는 곳
	// 3차원의 특정 평면(z=0)위에 있다고 가정하고 삼각형의 넓이를 외적으로 구해줌
	//m_data_pts 에서 첫 번째 점을 기준으로 각 점을 벡터로 구함
	std::vector<cv::Point3f> m_data_vec;
	std::vector<cv::Point3f> v1;
	//std::vector<cv::Point3f> v2;
	int x0_ = m_data_pts[0].x;
	int y0_ = m_data_pts[0].y;
	int pm; // plus minus 부호 정의 필요
	int x0, y0, z0, x1, y1, z1; // 외적으로 넓이 구할 때 좌표
	float area_sum = 0;
	float area_temp;
	std::vector<cv::Point3d> cross_product; // 외적 결과 vec

	// 벡터 만드는 for문하고 따로 area 구하기 vs 벡터 만들면서 area구하기
	for (int i = 1; i < m_data_pts.size(); i++){
		// vec 만드는 거 필요
		int x_ = m_data_pts[i].x;
		int y_ = m_data_pts[i].y;
		m_data_vec.push_back(Point3d(x_ - x0_, y_ - y0_, 1));
		if ((i>1)&(i<m_data_pts.size())){	// 여기서 외적해주기
			// 외적 함수를 못찾음..
			//printf("%d \n",i);
			x0 = int(m_data_vec[i - 2].x);
			y0 = int(m_data_vec[i - 2].y);
			z0 = int(m_data_vec[i - 2].z);
			//	printf("v0 : %d %d %d \n", x0,y0,z0);
			x1 = int(m_data_vec[i - 1].x);
			y1 = int(m_data_vec[i - 1].y);
			z1 = int(m_data_vec[i - 1].z);
			//	printf("v1 : %d %d %d \n", x1, y1, z1);
			cross_product.push_back(Point3d(y0*z1 - y1*z0, -x0*z1 + x1*z0, x0*y1 - x1*y0));
			if ((x0*y1 + y0*x1) >= 0){
				pm = 1;
			}
			else  pm = -1;
			//		printf("%f \n",area_sum);
			//area_sum += sqrt((cross_product[i - 2].x, cross_product[i - 2].y, cross_product[i - 2].z));
			area_temp = pm*sqrt(cross_product[i - 2].x*cross_product[i - 2].x + cross_product[i - 2].y*cross_product[i - 2].y + cross_product[i - 2].z*cross_product[i - 2].z);
			//	printf("temp : %f \n", area_temp);
			area_sum += area_temp;
			//		area_sum += cross_product[i - 2].x;
		}
	}
	printf("계산 완료\n");
	if (area_sum >= 0) return area_sum;
	else return -area_sum;
}

// return true if pt is interior point
bool PolygonDemo::ptInPolygon(const std::vector<cv::Point>& vtx, Point pt)
{
	return false;
}

// return homography type: NORMAL, CONCAVE, TWIST, REFLECTION, CONCAVE_REFLECTION
int PolygonDemo::classifyHomography(const std::vector<cv::Point>& pts1, const std::vector<cv::Point>& pts2)
{
	// 여기서 classifyHomography를 구현
	// 외적으로 점들의 방향 관계를 알 수 있음
	// 내적으로 각 각도의 예각/직각/둔각 추정할 수 있음
	// Concave 여부를 제일 먼저 분류
	//	cv::Mat cross_product = (cv::Mat_<int>(1,4)<<0,0,0,0);
	int n, result = 1, temp, cnt = 0;
	if (pts1.size() != 4 || pts2.size() != 4) return -1;	// pts1, pts2로 가능하나보네...ㅠㅠ
	// pts1 은 왼쪽에 뜨는 사각형, pts2가 내가 찍은 점 순서대로
	// 외적 구해주는 for문
	printf("check   %d  \n", pts2.size());
	for (int i = 0; i < pts2.size(); i++){
		if (i == 0) n = 3;
		else n = i - 1;
		std::vector<cv::Point> v1, v2;
		v1.push_back(cv::Point(pts2[n].x - pts2[i].x, pts2[n].y - pts2[i].y));
		v2.push_back(cv::Point(pts2[(i + 1) % 4].x - pts2[i].x, pts2[(i + 1) % 4].y - pts2[i].y));
		temp = v1[0].x*v2[0].y - v1[0].y*v2[0].x;
		//cross_product.at<int>(1, i) = temp;   // v1 x v2의 cross product // 얘 왜 에러뜨냐...
		printf("%d 번째 외적 : %d \n", i, temp);
		if (temp >= 0)temp = 1;
		if (temp < 0)temp = -1;
		result *= temp;
		if (temp < 0)cnt++;
	}
	//	printf("check result : %d cnt : %d\n",result,cnt);

	if (result >= 0){
		if (cnt == 0) return NORMAL;
		if (cnt == 2) return TWIST;
		if (cnt == 4) return REFLECTION;
	}
	else {
		if (cnt == 1) return CONCAVE;
		if (cnt == 3) return CONCAVE_REFLECTION;
	}

}

bool PolygonDemo::fitLine_axb(const std::vector<cv::Point>& pts, cv::Point2d& point_dep, cv::Point2d& point_arr)
{
	// 2019.04.29 line fitting
	int n = pts.size();
	// ax - y + b = 0 꼴로 근사
	// AX = B 로 구함 (nonhomogeneous) -> inverse 함수를 구하면 됨
	cv::Mat A = Mat::ones(n,2,CV_64F);
	cv::Mat B = Mat(n, 1, CV_64F);
	Mat A_inv, X;

	for (int i = 0; i < n; i++) {
		int x = pts[i].x;
		int y = pts[i].y;

		A.at<double>(i, 0) = x;
		B.at<double>(i, 0) = y;
	}	
	cv::invert(A, A_inv, DECOMP_SVD); // inverse 구해줌
	X = A_inv * B;

	double a = X.at<double>(0, 0);
	double b = X.at<double>(1, 0);
	point_dep.x = 0;
	point_dep.y = a * point_dep.x + b;
	point_arr.x = 640;
	point_arr.y = a * point_arr.x + b;
	return true;
}


bool PolygonDemo::fitLine_axby(const std::vector<cv::Point>& pts, cv::Point2d& point_dep, cv::Point2d& point_arr)
{
	// 2019.04.29 line fitting

	// ax + by + c = 0 꼴로 근사
	// 점을 Matrix 꼴로 바꿔주기
	//Point2d point_dep, point_arr;
	int n = (int)pts.size();
	cv::Mat A(pts.size(), 3, CV_64F);// size pts.size()*3으로
	cv::Mat w, u, v_t,sv;

	// Ax = 0 을 위해
	for (int i = 0; i<n; i++){
		A.at<double>(i, 0) = pts[i].x;
		A.at<double>(i, 1) = pts[i].y;
		A.at<double>(i, 2) = 1;
	}

	// PCA의 right singular vector구해야함
	SVD::compute(A,w,u,v_t,SVD::FULL_UV);
	sv = v_t.row(2);		// v_t의 마지막 행

	// axis, center에 대해서 저장

	//Point2d center_e;
	double a = sv.at<double>(0,0);
	double b = sv.at<double>(0,1);
	double c = sv.at<double>(0,2);
	printf("axby\n a: %f, b: %f, c: %f\n",a,b,c);
	point_dep.x = 0;
	point_dep.y = (-a*point_dep.x - c) / b;
	point_arr.x = 640;
	point_arr.y = (-a*point_arr.x - c) / b;

	return true;
}

// estimate a circle that best approximates the input points and return center and radius of the estimate circle
bool PolygonDemo::fitCircle(const std::vector<cv::Point>& pts, cv::Point2d& center, double& radius)
{
	// 점들을 vector-vector 꼴로 받아서 할지 Mat으로 할지 결정
	// a,b,c를 구하고 그것으로 원점 구하고, radius 구하는 식 구현 필요

	int n = (int)pts.size();
	if (n < 3) return false;

	// 점을 Matrix 꼴로 바꿔주기
	cv::Mat X(pts.size(), 3, CV_64F);// size pts.size()*3으로
	cv::Mat Y(pts.size(), 1, CV_64F);
	cv::Mat X_inv, X_, I, AX; // pseudo inverse of X

	// Ax = B를 위해 X가 A의미, Y가 B의미
	for (int i = 0; i<pts.size(); i++){
		X.at<double>(i, 0) = pts[i].x;
		X.at<double>(i, 1) = pts[i].y;
		X.at<double>(i, 2) = 1;
		Y.at<double>(i, 0) = -pts[i].x*pts[i].x - pts[i].y*pts[i].y;
	}

	// pseudo inverse를 구해야 함 X-랑 Y 곱하면 a,b,c 나옴
	//	cv::solve(X, Y, C, cv::DECOMP_SVD);
	cv::invert(X, X_inv, cv::DECOMP_SVD);
	X_ = X_inv*Y;

	std::vector<double> coef; // coefficient a,b,c
	X_.col(0).copyTo(coef);
	center.x = -0.5*coef[0];  // a,b,c 모두 구했기 때문에 그릴 수 있음
	center.y = -0.5*coef[1];
	radius = sqrt(0.25*coef[0] * coef[0] + 0.25*coef[1] * coef[1] - coef[2]);
	AX = X*X_;
	printf("(%f,%f)  r = %f\n", center.x, center.y, radius);
	return true;
}

bool PolygonDemo::fitEllipse(const std::vector<cv::Point>& pts, cv::Point2d& center_e, cv::Point2d& v){
	// 타원의 방정식은 결국 쌍곡선의 방정식의 한 형태
	int n = (int)pts.size();
	if (n < 4) return false;

	// 점을 Matrix 꼴로 바꿔주기
	cv::Mat A(pts.size(), 6, CV_64F);// size pts.size()*3으로
	cv::Mat Y(pts.size(), 1, CV_64F);
	cv::Mat A_SVD, X_, I, AX, sv; // pseudo inverse of X

	// Ax = 0 을 위해
	for (int i = 0; i<n; i++){
		A.at<double>(i, 0) = pts[i].x*pts[i].x;
		A.at<double>(i, 1) = pts[i].x*pts[i].y;
		A.at<double>(i, 2) = pts[i].y*pts[i].y;
		A.at<double>(i, 3) = pts[i].x;
		A.at<double>(i, 4) = pts[i].y;
		A.at<double>(i, 5) = 1;
	}
	cv::invert(A, A_SVD, cv::DECOMP_SVD); // A_SVD의 size는 A와 반대
	sv = A_SVD.col(n - 1); // SVD 결과 제일 오른쪽 열
	AX = A*sv;  // 0 확인
	// axis, center에 대해서 저장

	//Point2d center_e;
	double a = sv.at<double>(0, 0);
	double b = sv.at<double>(1, 0);
	double c = sv.at<double>(2, 0);
	double d = sv.at<double>(3, 0);
	double e = sv.at<double>(4, 0);
	double f = sv.at<double>(5, 0);
	printf("a : %f\nb : %f\nc : %f\nd : %f\ne : %f\nf : %f\n", a, b, c, d, e, f);
	printf("AX\n");
	cout << AX << endl;
	//cv::Size axes(size_x, size_y);
	return true;
}

void PolygonDemo::drawPolygon(Mat& frame, const std::vector<cv::Point>& vtx, bool closed)
{
	int i = 0;
	for (i = 0; i < (int)m_data_pts.size(); i++)
	{
		circle(frame, m_data_pts[i], 2, Scalar(255, 255, 255), CV_FILLED);
		// 좌표 출력
		char str[100];
		sprintf_s(str, 100, "(%d  %d)", m_data_pts[i].x, m_data_pts[i].y);
		putText(frame, str, Point(m_data_pts[i].x + 15, m_data_pts[i].y + 15), FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 255, 255), 1);
	}
	for (i = 0; i < (int)m_data_pts.size() - 1; i++)
	{
		//line(frame, m_data_pts[i], m_data_pts[i + 1], Scalar(255, 255, 255), 1);
	}
	if (closed)
	{
		//line(frame, m_data_pts[i], m_data_pts[0], Scalar(255, 255, 255), 1);
	}
}

void PolygonDemo::handleMouseEvent(int evt, int x, int y, int flags)
{
	if (evt == CV_EVENT_LBUTTONDOWN)
	{
		if (!m_data_ready)
		{
			m_data_pts.push_back(Point(x, y));
			printf("x : %d,  y : %d\n", x, y);
		}
		else
		{
			m_test_pts.push_back(Point(x, y));
		}
		refreshWindow();
	}
	else if (evt == CV_EVENT_LBUTTONUP)
	{
	}
	else if (evt == CV_EVENT_LBUTTONDBLCLK)
	{
		m_data_ready = true;
		refreshWindow();
	}
	else if (evt == CV_EVENT_RBUTTONDBLCLK)
	{
	}
	else if (evt == CV_EVENT_MOUSEMOVE)
	{
	}
	else if (evt == CV_EVENT_RBUTTONDOWN)
	{
		m_data_pts.clear();


		m_test_pts.clear();
		m_data_ready = false;
		refreshWindow();
	}
	else if (evt == CV_EVENT_RBUTTONUP)
	{
	}
	else if (evt == CV_EVENT_MBUTTONDOWN)
	{
	}
	else if (evt == CV_EVENT_MBUTTONUP)
	{
	}

	if (flags&CV_EVENT_FLAG_LBUTTON)
	{
	}
	if (flags&CV_EVENT_FLAG_RBUTTON)
	{
	}
	if (flags&CV_EVENT_FLAG_MBUTTON)
	{
	}
	if (flags&CV_EVENT_FLAG_CTRLKEY)
	{
	}
	if (flags&CV_EVENT_FLAG_SHIFTKEY)
	{
	}
	if (flags&CV_EVENT_FLAG_ALTKEY)
	{
	}
}








