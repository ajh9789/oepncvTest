#include <opencv2/imgproc.hpp> 
#include <opencv2/highgui.hpp>
#include <iostream> 
using namespace cv;
using namespace std;

int main() {
Mat img = imread("Lenna.png", 1);//이미지 읽기
if (img.empty()) { //이미지 비어있으면
	cerr << "Error: Could not load image!" << endl; //예외 처리
	return -1;//종료
}
// 1. 그레이스케일 변환
cv::Mat gray;
cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

// 2. 컬러맵 적용 (흑백)
cv::Mat xray;
cv::applyColorMap(gray, xray, cv::COLORMAP_BONE);

// 3. 이미지 출력
cv::imshow("Original", img);
cv::imshow("X-ray Style", xray);
cv::waitKey(0);

return 0;
}

