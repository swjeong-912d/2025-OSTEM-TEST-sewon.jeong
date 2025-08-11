#include <iostream>
#include <cstdint>
#include <opencv2/opencv.hpp>

using std::uint64_t;


static uint64_t Solution(uint64_t A);


int main()
{
	Solution(0);
	return 0;
}
//========//========//========//========//=======#//========//========//========//========//=======#


uint64_t Solution(uint64_t A)
{
	std::string base_dir = "../../tooth_segmentation/imgs/";
	cv::Mat image = cv::imread(base_dir + "1.png");
	if (image.empty()) {
		std::cerr << "Error: Could not load the image!" << std::endl;
		return -1;
	}

	// Convert the image from BGR to LAB color space
	cv::Mat lab_image;
	cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);

	// Display the LAB image
	cv::imshow("LAB Image", lab_image);
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}