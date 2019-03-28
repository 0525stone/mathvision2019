#include "polygon_demo.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

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
		// ���� ������ִ� ��Ʈ
		if (m_param.compute_area)
		{
			int area = polyArea(m_data_pts);
			char str[100];
			sprintf_s(str, 100, "Area = %d", area);
			putText(frame, str, Point(25, 25), FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
			printf("��� �Ϸ�\n");
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

			// check homography  classifyHomography�� ���淡 �̰��� ���(homo_type)�� �ٷ� ���¸� ����������..
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
			// ���⵵ �� ����.
			// Ÿ���� �׸��� ���ؼ� ��� �ʿ��Ѱ�...? - 5���� ��� �ְ�� ���������� Ǫ�� �Ŵϱ�
			Point2d center_e;
			Point2d v;	// v�� ������ ���� ����
			double angle = 0, startAngle = 0, endAngle = 360;
			int  arcStart = 0, arcEnd = 0, delta = 0;
			RotatedRect rec_e; // center , size, angle �� ��µ� (ellipse�� �ʿ��� ����)
			// ���� : ellipse(frame,center,Size axes, double angle, double startAngle, double endAngle,color, thickness,linetype,shift)
			bool ok = fitEllipse(m_data_pts, center_e, v);
			std::vector<cv::Point> points_e;
			float axes_scale = 0.5;
		//	bool ok = true;
			if (ok)
			{
				//printf("center of ellipse :(%f %f)\naxes: %f %f\n", center_e.x, center_e.y, v.x, v.y);
				rec_e = cv::fitEllipse(m_data_pts);  // rectangle �� ������ش�
				//ellipse2Poly(rec_e.center, rec_e.size, rec_e.angle, arcStart, arcEnd, delta, points_e); //vector<Point>& pts
				printf("done\n");
				cv::Size axes(rec_e.size.width*axes_scale, rec_e.size.height * axes_scale);
				ellipse(frame, rec_e.center, axes, rec_e.angle, startAngle, endAngle, Scalar(0, 255, 255), 1);
			}
		}
	}

	imshow("PolygonDemo", frame);
}

// return the area of polygon
int PolygonDemo::polyArea(const std::vector<cv::Point>& vtx)
{
	// ���Ⱑ ���� �����ִ� ��
	// 3������ Ư�� ���(z=0)���� �ִٰ� �����ϰ� �ﰢ���� ���̸� �������� ������
	//m_data_pts ���� ù ��° ���� �������� �� ���� ���ͷ� ����
	std::vector<cv::Point3f> m_data_vec;
	std::vector<cv::Point3f> v1;
	//std::vector<cv::Point3f> v2;
	int x0_ = m_data_pts[0].x;
	int y0_ = m_data_pts[0].y;
	int pm; // plus minus ��ȣ ���� �ʿ�
	int x0, y0, z0, x1, y1, z1; // �������� ���� ���� �� ��ǥ
	float area_sum = 0;
	float area_temp;
	std::vector<cv::Point3d> cross_product; // ���� ��� vec

	// ���� ����� for���ϰ� ���� area ���ϱ� vs ���� ����鼭 area���ϱ�
	for (int i = 1; i < m_data_pts.size(); i++){
		// vec ����� �� �ʿ�
		int x_ = m_data_pts[i].x;
		int y_ = m_data_pts[i].y;
		m_data_vec.push_back(Point3d(x_ - x0_, y_ - y0_, 1));
		if ((i>1)&(i<m_data_pts.size())){	// ���⼭ �������ֱ�
			// ���� �Լ��� ��ã��..
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
	printf("��� �Ϸ�\n");
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
	// ���⼭ classifyHomography�� ����
	// �������� ������ ���� ���踦 �� �� ����
	// �������� �� ������ ����/����/�а� ������ �� ����
	// Concave ���θ� ���� ���� �з�
	//	cv::Mat cross_product = (cv::Mat_<int>(1,4)<<0,0,0,0);
	int n, result = 1, temp, cnt = 0;
	if (pts1.size() != 4 || pts2.size() != 4) return -1;	// pts1, pts2�� �����ϳ�����...�Ф�
	// pts1 �� ���ʿ� �ߴ� �簢��, pts2�� ���� ���� �� �������
	// ���� �����ִ� for��
	printf("check   %d  \n", pts2.size());
	for (int i = 0; i < pts2.size(); i++){
		if (i == 0) n = 3;
		else n = i - 1;
		std::vector<cv::Point> v1, v2;
		v1.push_back(cv::Point(pts2[n].x - pts2[i].x, pts2[n].y - pts2[i].y));
		v2.push_back(cv::Point(pts2[(i + 1) % 4].x - pts2[i].x, pts2[(i + 1) % 4].y - pts2[i].y));
		temp = v1[0].x*v2[0].y - v1[0].y*v2[0].x;
		//cross_product.at<int>(1, i) = temp;   // v1 x v2�� cross product // �� �� �����߳�...
		printf("%d ��° ���� : %d \n", i, temp);
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

// estimate a circle that best approximates the input points and return center and radius of the estimate circle
bool PolygonDemo::fitCircle(const std::vector<cv::Point>& pts, cv::Point2d& center, double& radius)
{
	// ������ vector-vector �÷� �޾Ƽ� ���� Mat���� ���� ����
	// a,b,c�� ���ϰ� �װ����� ���� ���ϰ�, radius ���ϴ� �� ���� �ʿ�

	int n = (int)pts.size();
	if (n < 3) return false;

	// ���� Matrix �÷� �ٲ��ֱ�
	cv::Mat X(pts.size(), 3, CV_64F);// size pts.size()*3����
	cv::Mat Y(pts.size(), 1, CV_64F);
	cv::Mat X_inv, X_, I, AX; // pseudo inverse of X

	// Ax = B�� ���� X�� A�ǹ�, Y�� B�ǹ�
	for (int i = 0; i<pts.size(); i++){
		X.at<double>(i, 0) = pts[i].x;
		X.at<double>(i, 1) = pts[i].y;
		X.at<double>(i, 2) = 1;
		Y.at<double>(i, 0) = -pts[i].x*pts[i].x - pts[i].y*pts[i].y;
	}

	// pseudo inverse�� ���ؾ� �� X-�� Y ���ϸ� a,b,c ����
	//	cv::solve(X, Y, C, cv::DECOMP_SVD);
	cv::invert(X, X_inv, cv::DECOMP_SVD);
	X_ = X_inv*Y;

	std::vector<double> coef; // coefficient a,b,c
	X_.col(0).copyTo(coef);
	center.x = -0.5*coef[0];  // a,b,c ��� ���߱� ������ �׸� �� ����
	center.y = -0.5*coef[1];
	radius = sqrt(0.25*coef[0] * coef[0] + 0.25*coef[1] * coef[1] - coef[2]);
	AX = X*X_;
	printf("(%f,%f)  r = %f\n", center.x, center.y, radius);
	return true;
}
bool PolygonDemo::fitEllipse(const std::vector<cv::Point>& pts, cv::Point2d& center_e, cv::Point2d& v){
	// Ÿ���� �������� �ᱹ �ְ�� �������� �� ����
	int n = (int)pts.size();
	if (n < 4) return false;

	// ���� Matrix �÷� �ٲ��ֱ�
	cv::Mat A(pts.size(), 6, CV_64F);// size pts.size()*3����
	cv::Mat Y(pts.size(), 1, CV_64F);
	cv::Mat A_SVD, X_, I, AX, sv; // pseudo inverse of X

	// Ax = 0 �� ����
	for (int i = 0; i<n; i++){
		A.at<double>(i, 0) = pts[i].x*pts[i].x;
		A.at<double>(i, 1) = pts[i].x*pts[i].y;
		A.at<double>(i, 2) = pts[i].y*pts[i].y;
		A.at<double>(i, 3) = pts[i].x;
		A.at<double>(i, 4) = pts[i].y;
		A.at<double>(i, 5) = 1;
	}
	cv::invert(A, A_SVD, cv::DECOMP_SVD); // A_SVD�� size�� A�� �ݴ�
	sv = A_SVD.col(n - 1); // SVD ��� ���� ������ ��
	AX = A*sv;  // 0 Ȯ��
	// axis, center�� ���ؼ� ����

	//Point2d center_e;
	double a = sv.at<double>(0, 0);
	double b = sv.at<double>(1, 0);
	double c = sv.at<double>(2, 0);
	double d = sv.at<double>(3, 0);
	double e = sv.at<double>(4, 0);
	double f = sv.at<double>(5, 0);
	printf("a : %f\nb : %f\nc : %f\nd : %f\ne : %f\nf : %f\n",a,b,c,d,e,f);
	printf("AX\n");
	cout << AX<< endl;
	//cv::Size axes(size_x, size_y);
	return true;
}
void PolygonDemo::drawPolygon(Mat& frame, const std::vector<cv::Point>& vtx, bool closed)
{
	int i = 0;
	for (i = 0; i < (int)m_data_pts.size(); i++)
	{
		circle(frame, m_data_pts[i], 2, Scalar(255, 255, 255), CV_FILLED);
		// ��ǥ ���
		char str[100];
		sprintf_s(str, 100, "(%d  %d)", m_data_pts[i].x, m_data_pts[i].y);
		putText(frame, str, Point(m_data_pts[i].x + 15, m_data_pts[i].y + 15), FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 255, 255), 1);
	}
	for (i = 0; i < (int)m_data_pts.size() - 1; i++)
	{
		line(frame, m_data_pts[i], m_data_pts[i + 1], Scalar(255, 255, 255), 1);
	}
	if (closed)
	{
		line(frame, m_data_pts[i], m_data_pts[0], Scalar(255, 255, 255), 1);
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
