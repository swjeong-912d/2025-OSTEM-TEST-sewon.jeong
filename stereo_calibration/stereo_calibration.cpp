#include <iostream>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <numeric>

using std::uint64_t;


static uint64_t Solution(uint64_t A);
std::vector<cv::Point2f> computeCircleCenterCoordinates(const cv::Mat& image);
std::vector<cv::Point2f> findMostIsolatedPointCoordinates(const std::vector<cv::Point2f>& keyPointCoordinates, size_t K = 5);

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

		std::vector<cv::Point2f> keyPointCoordinates = computeCircleCenterCoordinates(image); // Detect keypoints
		auto selectedKeypointCoordinates = findMostIsolatedPointCoordinates(keyPointCoordinates, 5); // Find the most isolated keypoints

		//sort the keypoints based on their coordinates
		size_t W = image.size().width;
		std::sort(selectedKeypointCoordinates.begin(), selectedKeypointCoordinates.end(), [&W](const cv::Point2f& a, const cv::Point2f& b) {
			return a.x + W * a.y < b.x + W * b.y;
			});

		// Display the image with the furthest keypoints marked
		cv::Mat outputImage = image.clone();
		for (const auto& coordinate: selectedKeypointCoordinates)
		{
			cv::circle(outputImage, coordinate, 5, cv::Scalar(0, 255, 0), -1); // Draw a circle around each keypoint
		}
		cv::imshow("Furthest Keypoints", outputImage); // Show the image with keypoints
		cv::waitKey(0); // Wait for a key press to close the window
		cv::destroyAllWindows(); // Close all OpenCV windows

		// Then, extract the five key points with the furthest distance.
	}
	return 0;
}

std::vector<cv::Point2f> computeCircleCenterCoordinates(const cv::Mat& image)
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
	std::vector<cv::Point2f> keypointCoords;
	for (const auto& keypoint : keypoints)
	{
		keypointCoords.push_back(keypoint.pt);
	}

	return keypointCoords;
}

std::vector<cv::Point2f> findMostIsolatedPointCoordinates(const std::vector<cv::Point2f>& keyPointCoordinates, size_t K)
{
	// First, try searching with a loop. If it's slow, try a tree.
	std::vector<double> closestNeighborDists;
	std::vector<double> neighborDists;
	closestNeighborDists.reserve(keyPointCoordinates.size());
	neighborDists.reserve(keyPointCoordinates.size() - 1);
	for (const auto& keypointCoord: keyPointCoordinates)
	{
		neighborDists.clear();
		for (const auto& neighborCoord : keyPointCoordinates)
		{
			if (neighborCoord == keypointCoord) // You can search by address, but for readability...
			{
				continue;
			}

			neighborDists.push_back(cv::norm(neighborCoord - keypointCoord));
		}

		closestNeighborDists.push_back(*std::min_element(neighborDists.begin(), neighborDists.end()));
	}


	std::vector<int> idx(closestNeighborDists.size());
	std::iota(idx.begin(), idx.end(), 0);

	// Sort corresponding indexes in descending order of distance value
	std::sort(idx.begin(), idx.end(), [&closestNeighborDists](int a, int b) {
			return closestNeighborDists[a] > closestNeighborDists[b];
		});

	std::vector<cv::Point2f> outputKeyPointCoords;
	outputKeyPointCoords.reserve(K);


	size_t topN = std::min(keyPointCoordinates.size(), K);
	for (int i = 0; i < topN; i++)
	{
		int id = idx[i];
		outputKeyPointCoords.push_back(keyPointCoordinates[id]);
	}

	return outputKeyPointCoords; // Return the K most isolated keypoints
}