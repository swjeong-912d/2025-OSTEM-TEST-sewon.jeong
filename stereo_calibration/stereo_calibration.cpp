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
	// 1. Image path setting
	std::vector<cv::Mat> images;
	std::string base_dir = "../../stereo_calibration/imgs/";
	std::string imageNames = "*.bmp";
	std::vector<cv::String> fileNames;
	cv::glob(base_dir + imageNames, fileNames);


	// 2. Read images and process each image
	for (const auto& fileName : fileNames) 
	{
		cv::Mat image = cv::imread(fileName, cv::IMREAD_COLOR); // Read the image
		if (image.empty()) 
		{
			std::cerr << "Error: Could not load image " << fileName << std::endl;
			continue; // Skip to the next file if loading fails
		}

		// 3. Blob Detector parameters
		cv::SimpleBlobDetector::Params params;
		params.filterByColor = true;
		params.blobColor = 255; // 0: dark blob, 255: bright blob

		params.filterByArea = true;
		params.minArea = 3*3; // px^2
		params.maxArea = 30*30; // px^2

		params.filterByCircularity = true;
		params.minCircularity = 0.8; // 0~1

		params.filterByInertia = true;
		params.minInertiaRatio = 0.5; // 0~1

		params.filterByConvexity = true;
		params.minConvexity = 0.8; // 0~1

		// 4. Create a SimpleBlobDetector with the parameters
		auto detector = cv::SimpleBlobDetector::create(params);

		// 5. Detect keypoints
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(image, keypoints);

		// 6. Draw keypoints on the image
		cv::Mat output;
		cv::drawKeypoints(image, keypoints, output, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		// 7. Print center coordinates of the keypoints
		for (size_t i = 0; i < keypoints.size(); i++) 
		{
			std::cout << "Center " << i << ": ("
				<< keypoints[i].pt.x << ", " << keypoints[i].pt.y << ")" << std::endl;
		}

		cv::imshow("Detected Circles", output);

		// Display image
		cv::waitKey(0);
		cv::destroyAllWindows();

		// TODO: Measure the distance to the nearest neighbor for each key point using FLANN. 
		// Then, extract the five key points with the furthest distance.
	}
	return 0;
}