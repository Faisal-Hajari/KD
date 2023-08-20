#include <opencv2/opencv.hpp>
#include <iostream>

bool isBlurred(const cv::Mat& image, double threshold = 100.0) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);

    return stddev.val[0] < threshold;
}

int main() {
    cv::Mat image = cv::imread("your_image.jpg");
    if (image.empty()) {
        std::cerr << "Image not found or invalid." << std::endl;
        return 1;
    }

    bool blurry = isBlurred(image);
    std::cout << "Image is blurry: " << std::boolalpha << blurry << std::endl;

    return 0;
}