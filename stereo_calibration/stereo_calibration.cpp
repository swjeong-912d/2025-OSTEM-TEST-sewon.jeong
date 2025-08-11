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
	std::string base_dir = "../../stereo_calibration/imgs/";
	cv::Mat image = cv::imread(base_dir + "1.bmp");
	if (image.empty()) 
	{
		std::cerr << "Error: Could not load the image!" << std::endl;
		return -1;
	}

	// Display image
	cv::imshow("Calibration Image", image);
	cv::waitKey(0);
	cv::destroyAllWindows();
	
	return 0;
}