#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>  
#include <opencv2/core/utils/filesystem.hpp> 
#include <iostream>
#include <filesystem>
using namespace cv;
using namespace std;
namespace fs = std::filesystem;

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
        if (area > 1000 && area < 100000) { // 결함의 크기 결정
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

// 템플릿 매칭 함수
void runTemplateMatching(const string& templatePath, const string& targetPath) {
    Mat templ = imread(templatePath, IMREAD_GRAYSCALE);
    Mat target = imread(targetPath, IMREAD_GRAYSCALE);
    if (templ.empty() || target.empty()) {
        cerr << "이미지 로딩 실패" << endl;
        return;
    }

    Mat result;
    // 결과 크기 계산
    int result_cols = target.cols - templ.cols + 1;
    int result_rows = target.rows - templ.rows + 1;
    result.create(result_rows, result_cols, CV_32FC1);

    // 템플릿 매칭 실행
    matchTemplate(target, templ, result, TM_CCOEFF_NORMED);
    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

    // 가장 유사한 위치 찾기
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    cout << "유사도 (TM_CCOEFF_NORMED): " << maxVal << endl;
    if (maxVal < 0.7) {
        cout << "⚠️ 불량 가능성 있음!" << endl;
	}
	else {
		cout << "✅ 양품!" << endl;
	}
    // 시각화
    Point matchLoc = maxLoc;
    rectangle(target, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 0, 255), 2);
    imshow("Template Match Result", target);
    waitKey(0);
}

// 이미지 처리 및 저장 함수
void processAndSave(const string& path, const string& outputName) {
    Mat img;
    if (!loadImage(path, img)) return;

    Mat filtered = applyXrayFilter(img);
    vector<vector<Point>> contours;
    Mat result = detectContours(filtered, contours);
    drawDefects(result, contours);

    size_t pos = path.find('/'); // '/'의 위치를 찾음
    string basePath = (pos != string::npos) ? path.substr(0, pos) : path; // '/' 이전 부분 추출
    if (!fs::exists(basePath + "/results")) { // results 폴더가 없으면 생성
        fs::create_directory(basePath + "/results"); // 기존 경로에 results 폴더 생성
    }
    string outputPath = basePath + "/results/result_" + outputName; // 'result_' 접두사 추가
    imwrite(outputPath, result); //결과 저장
    cout << "Processed: " << path << " → " << outputPath << endl;
}

// ✅ ONNX 추론 함수: 이미지 1장 예측
int predictFromONNX(dnn::Net& net, const string& imagePath) {
    Mat img = imread(imagePath);
    if (img.empty()) {
        cerr << "이미지 로딩 실패: " << imagePath << endl;
        return -1;
    }

    Mat blob = dnn::blobFromImage(img, 1.0 / 255.0, Size(224, 224), Scalar(), true, false);
    net.setInput(blob);
    Mat output = net.forward();

    Point classIdPoint;
    double confidence;
    minMaxLoc(output, 0, &confidence, 0, &classIdPoint);
    return classIdPoint.x; // 예측 클래스 인덱스
}

//  전체 test/good 폴더 대상 ONNX 테스트 함수
void testONNXModel() {
    dnn::Net net = dnn::readNetFromONNX("hazel_model.onnx");

    vector<string> testImages;
    utils::fs::glob("hazelnut/test/good", "*.png", testImages);  // good 폴더만

    int correct = 0, total = 0;
    for (const auto& path : testImages) {
        int pred = predictFromONNX(net, path);
        total++;
        cout << "[" << total << "] " << path << " → 예측 클래스: " << pred << endl;
        if (pred == 2) correct++; // {'crack': 0, 'cut': 1, 'good': 2, 'hole': 3, 'print': 4}
    }

    cout << "\n✅ ONNX 모델 정확도 (good): " << (correct * 100.0 / total) << "% (" << correct << "/" << total << ")" << endl;
}

//  main 함수에 테스트 호출 추가
int main(int argc, char** argv) {
    //// 템플릿 매칭 테스트
    //runTemplateMatching("hazelnut/train/good/001.png", "hazelnut/test/cut/003.png");

    //// 이미지 처리 및 저장
    //processAndSave("hazelnut/test/good/001.png", "good_001.png");
    //processAndSave("hazelnut/test/crack/002.png", "crack_002.png");
    //processAndSave("hazelnut/test/hole/003.png", "hole_003.png");
    //processAndSave("hazelnut/test/cut/004.png", "cut_004.png");
    //processAndSave("hazelnut/test/print/005.png", "print_005.png");

    //  ONNX 모델 테스트
    testONNXModel();

    waitKey(0);
    return 0;
}