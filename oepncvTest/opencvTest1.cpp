#include <opencv2/imgproc.hpp> 
#include <opencv2/highgui.hpp>
#include <iostream> 
using namespace cv;
using namespace std;

int main() {
    // 이미지 읽기
    Mat img = imread("Lenna.png", 1);
    if (img.empty()) { // 이미지 비어있으면 예외 처리
        cerr << "Error: Could not load image!" << endl;
        return -1; // 종료
    }

    // 1. 그레이스케일 변환
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // 2. 가우시안 블러 적용 (노이즈 제거를 위해)
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // 3. 이진화 (thresholding)
    cv::Mat binary;
    cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 4. 외곽선 찾기
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // 5. 외곽선 필터링 (너무 작은 외곽선 제거) 외곽선 정확도 올림
    vector<vector<cv::Point>> filteredContours;
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) > 100) { // 면적이 100 이상인 외곽선만 사용
            filteredContours.push_back(contour);
        }
    }
    // 6. 원본 이미지에 외곽선 그리기
    cv::Mat result = img.clone();
    cv::drawContours(result, filteredContours, -1, cv::Scalar(0, 0, 255), 2); // 빨간색 외곽선

    // 7. 결과 보여주기
    cv::imshow("Contour Detection", result);
    cv::waitKey(0);

    return 0;
}
