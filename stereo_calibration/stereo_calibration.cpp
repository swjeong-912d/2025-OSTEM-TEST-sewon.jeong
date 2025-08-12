#include <iostream>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <numeric>

using std::uint64_t;


static uint64_t Solution(uint64_t A);
std::vector<cv::KeyPoint> computeCircleCenterCoordinates(const cv::Mat& image);
std::vector<cv::KeyPoint> findMostIsolatedPoints(const std::vector<cv::KeyPoint>& keyPoints, size_t K = 5);

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

		std::vector<cv::KeyPoint> keypoints = computeCircleCenterCoordinates(image); // Detect keypoints
		auto selectedKeypoints = findMostIsolatedPoints(keypoints, 5); // Find the most isolated keypoints

		// Display the image with the furthest keypoints marked

		cv::Mat outputImage = image.clone();
		for (const auto& keypoint : selectedKeypoints) 
		{
			cv::circle(outputImage, keypoint.pt, 5, cv::Scalar(0, 255, 0), -1); // Draw a circle around each keypoint
		}
		cv::imshow("Furthest Keypoints", outputImage); // Show the image with keypoints
		cv::waitKey(0); // Wait for a key press to close the window
		cv::destroyAllWindows(); // Close all OpenCV windows

		// Then, extract the five key points with the furthest distance.
	}
	return 0;
}

std::vector<cv::KeyPoint> computeCircleCenterCoordinates(const cv::Mat& image) 
{
	// Blob Detector parameters
	cv::SimpleBlobDetector::Params params;
	params.filterByColor = true;
	params.blobColor = 255; // 0: dark blob, 255: bright blob

	params.filterByArea = true;
	params.minArea = 3 * 3; // px^2
	params.maxArea = 30 * 30; // px^2

	params.filterByCircularity = true;
	params.minCircularity = 0.8; // 0~1

	params.filterByInertia = true;
	params.minInertiaRatio = 0.5; // 0~1

	params.filterByConvexity = true;
	params.minConvexity = 0.8; // 0~1

	// Create a SimpleBlobDetector with the parameters
	auto detector = cv::SimpleBlobDetector::create(params);

	// Detect keypoints
	std::vector<cv::KeyPoint> keypoints;
	detector->detect(image, keypoints);

	return keypoints;
}

std::vector<cv::KeyPoint> findMostIsolatedPoints(const std::vector<cv::KeyPoint>& keypoints, size_t K) 
{
	// First, try searching with a loop. If it's slow, try a tree.
	std::vector<double> closestNeighborDists;
	std::vector<double> neighborDists;
	closestNeighborDists.reserve(keypoints.size());
	neighborDists.reserve(keypoints.size() - 1);
	for (const auto& keypoint : keypoints)
	{
		neighborDists.clear();
		for (const auto& neighbor : keypoints)
		{
			if (neighbor.pt == keypoint.pt) // You can search by address, but for readability...
			{
				continue;
			}

			neighborDists.push_back(cv::norm(neighbor.pt - keypoint.pt));
		}

		closestNeighborDists.push_back(*std::min_element(neighborDists.begin(), neighborDists.end()));
	}


	std::vector<int> idx(closestNeighborDists.size());
	std::iota(idx.begin(), idx.end(), 0);

	// Sort corresponding indexes in descending order of distance value
	std::sort(idx.begin(), idx.end(), [&closestNeighborDists](int a, int b) {
			return closestNeighborDists[a] > closestNeighborDists[b];
		});

	std::vector<cv::KeyPoint> outputKeypoints;
	outputKeypoints.reserve(K);


	size_t topN = std::min(keypoints.size(), K);
	for (int i = 0; i < topN; i++)
	{
		int id = idx[i];
		outputKeypoints.push_back(keypoints[id]);
	}

	return outputKeypoints; // Return the K most isolated keypoints
}