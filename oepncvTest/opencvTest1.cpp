
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
imshow("Lenna", img); //이미지 출력
waitKey(0);//키 입력 대기
return 0;
}

