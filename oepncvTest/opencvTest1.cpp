#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 이미지 로딩 함수
bool loadImage(const string& path, Mat& out) {
    out = imread(path, 1);
    if (out.empty()) {
        cerr << "Error: Could not load image!" << endl;
        return false;
    }
    return true;
}

// X-ray 스타일 필터 적용 함수
Mat applyXrayFilter(const Mat& img) {
    Mat gray, blurred, colored;
    cvtColor(img, gray, COLOR_BGR2GRAY);                      // 그레이스케일
    GaussianBlur(gray, blurred, Size(5, 5), 0);                // 가우시안 블러
    applyColorMap(blurred, colored, COLORMAP_BONE);           // X-ray 느낌 컬러맵
    return colored;
}

// 외곽선 검출 함수
Mat detectContours(const Mat& input, vector<vector<Point>>& filteredContours) {
    Mat gray, blurred, binary;

    // 그레이 + 블러 + 이진화 (thresholding)
    cvtColor(input, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5, 5), 0);
    threshold(blurred, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // Contour 찾기
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    // 필터링: 너무 작은 외곽선 제거
    for (const auto& contour : contours) {
        if (contourArea(contour) > 100) {
            filteredContours.push_back(contour);
        }
    }

    // 복사한 이미지에 외곽선 그림
    Mat result = input.clone();
    drawContours(result, filteredContours, -1, Scalar(0, 0, 255), 2); // 빨간색
    return result;
}

// 결함 위치 표시 함수
void drawDefects(Mat& img, const vector<vector<Point>>& contours) {
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area < 300) { // 작은 결함만 표시
            Moments m = moments(contour);
            if (m.m00 != 0) {
                int cx = int(m.m10 / m.m00);
                int cy = int(m.m01 / m.m00);
                circle(img, Point(cx, cy), 5, Scalar(0, 255, 255), -1); // 노란 점
                putText(img, "Defect", Point(cx + 10, cy), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            }
        }
    }
}

// 이미지 처리 및 저장 함수
void processAndSave(const string& path, const string& outputName) { 
    Mat img;
    if (!loadImage(path, img)) return;

    Mat filtered = applyXrayFilter(img);
    vector<vector<Point>> contours;
    Mat result = detectContours(filtered, contours);
    drawDefects(result, contours);

    imwrite(outputName, result); // 결과 저장
    cout << "Processed: " << path << " → " << outputName << endl;
}

// 메인 함수: 흐름만 조절
int main(int argc, char** argv) { //C++ 기본 main함수에서 argc, argv를 사용 전자는 명령줄 인자 개수, 후자는 인자 배열
    string path = (argc > 1) ? argv[1] : "Lenna.png";

    Mat img;
    if (!loadImage(path, img)) return -1;

    Mat filtered = applyXrayFilter(img);

    vector<vector<Point>> contours;
    Mat result = detectContours(filtered, contours);

    drawDefects(result, contours);
    //부분 테스트
    processAndSave("transistor/test/good/001.png", "result_good_001.png");
    processAndSave("transistor/test/cut_lead/002.png", "result_cut_lead_002.png");
    processAndSave("transistor/test/misalignment/003.png", "result_misalign_003.png");
    waitKey(0);

    return 0;
}