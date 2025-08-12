#include <iostream>
#include <cstdint>
#include <opencv2/opencv.hpp>

using std::uint64_t;


static uint64_t Solution(uint64_t A);
cv::Mat computeTeethMask(const cv::Mat& image);

int main()
{
	Solution(0);
	return 0;
}
//========//========//========//========//=======#//========//========//========//========//=======#


uint64_t Solution(uint64_t A)
{
	std::string base_dir = "../../tooth_segmentation/imgs/";
	std::string imageNames = "*.png";
	std::vector<cv::String> fileNames;
	cv::glob(base_dir + imageNames, fileNames);

	// 2. Read images and process each image
	for (const auto& fileName : fileNames)
	{
		cv::Mat image = cv::imread(fileName, cv::IMREAD_COLOR); // Read the image
		if (image.empty())
		{
			std::cerr << "Error: Could not load image " << fileName << std::endl;
		}
		else
		{
			std::cout << "Image load completed: " << fileName << std::endl;
		}

		cv::imshow("Image", image);
		cv::moveWindow("Image", 100, 100);

		cv::waitKey(0);
		cv::destroyAllWindows();

		cv::Mat maskedImage = cv::Mat::zeros(image.size(), image.type());
		auto mask = computeTeethMask(image);

		std::cout << "Teeth segmentation completed" << std::endl;
		std::cout << "Teeth region ratio: " << (100.0 * cv::countNonZero(mask)) / image.size().area() << "%" << std::endl;

		image.copyTo(maskedImage, mask);  // non-red aread
		cv::imshow("Segmented Tooth", maskedImage);
		cv::moveWindow("Segmented Tooth", 100, 100);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
	return 0;
}


cv::Mat computeTeethMask(const cv::Mat& image)
{
	cv::Mat shifted;
	cv::pyrMeanShiftFiltering(image, shifted, 20, 40);

	cv::Mat hsv;
	cv::cvtColor(shifted, hsv, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> channels;
	cv::split(hsv, channels);

	// binary mask for bright areas
	cv::Mat brightMask;
	cv::threshold(channels[2], brightMask, 110, 255, cv::THRESH_BINARY);

	// generate red mask (Hue: 0~10 deg, 160~180 deg)
	cv::Mat lowerRedMask, upperRedMask, redMask;
	cv::inRange(hsv, cv::Scalar(0, 50, 50), cv::Scalar(10, 255, 255), lowerRedMask);
	cv::inRange(hsv, cv::Scalar(160, 50, 50), cv::Scalar(180, 255, 255), upperRedMask);
	cv::bitwise_or(lowerRedMask, upperRedMask, redMask);

	// remove red areas from brightMask
	cv::Mat toothMask;
	cv::bitwise_and(brightMask, ~redMask, toothMask);

	// smooth the mask
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::morphologyEx(toothMask, toothMask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
	cv::morphologyEx(toothMask, toothMask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);

	return toothMask;
}