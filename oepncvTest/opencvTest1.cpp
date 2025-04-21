#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 이미지 읽기
    Mat img = imread("Lenna.png", 1);
    if (img.empty()) {
        cerr << "Error: Could not load image!" << endl;
        return -1;
    }

    // 1. 그레이스케일 변환
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // 2. 가우시안 블러 (노이즈 제거)
    Mat blurred;
    GaussianBlur(gray, blurred, Size(5, 5), 0);

    // 3. 자동 이진화 (OTSU 사용)
    Mat binary;
    threshold(blurred, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // 4. 외곽선 찾기
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    // 5. 외곽선 필터링 (작은 면적 제거)
    vector<vector<Point>> filteredContours;
    for (const auto& contour : contours) {
        if (contourArea(contour) > 100) {
            filteredContours.push_back(contour);
        }
    }

    // 6. 컬러맵 (X-ray 스타일) 적용
    Mat xray;
    applyColorMap(gray, xray, COLORMAP_BONE);  // X-ray 느낌

    // 7. 외곽선 + 결함 시각화
    Mat result = xray.clone();
    drawContours(result, filteredContours, -1, Scalar(0, 0, 255), 2); // 빨간 외곽선

    // 8. 작은 결함 (특정 크기 이하) 위치에 표시
    for (const auto& contour : filteredContours) {
        double area = contourArea(contour);
        if (area < 300) {
            Moments m = moments(contour);
            if (m.m00 != 0) {
                int cx = int(m.m10 / m.m00);
                int cy = int(m.m01 / m.m00);
                circle(result, Point(cx, cy), 5, Scalar(0, 255, 255), -1); // 노란 점
                putText(result, "Defect", Point(cx + 10, cy), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            }
        }
    }

    // 9. 결과 출력
    imshow("Original", img);
    imshow("X-ray Style + Defect", result);
    waitKey(0);

    return 0;
}