#include <iostream>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <numeric>

using std::uint64_t;


static uint64_t Solution(uint64_t A);
std::vector<cv::Point2f> computeCircleCenterCoordinates(const cv::Mat& image);
std::vector<cv::Point2f> findKNearestNeighborCoords(const cv::Point2f& queryPoint, const std::vector<cv::Point2f>& points, int K);
std::vector<cv::Point2f> findMostIsolatedPointGroupCoordinates(const std::vector<cv::Point2f>& keyPointCoordinates, size_t K = 5, size_t numNeighbors = 4);

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

	std::vector<cv::Point3f> pattern3D = {
		cv::Point3f(0,1,0),
		cv::Point3f(-1,0,0),
		cv::Point3f(0,0,0),
		cv::Point3f(1,0,0),
		cv::Point3f(0,-1,0) }; // 3D points in the calibration pattern

	std::vector<std::vector<cv::Point2f>> imagePoints; // Store 2D points from images
	std::vector<std::vector<cv::Point3f>> objectPoints;

	cv::Size imgSize;
	// 2. Read images and process each image
	for (const auto& fileName : fileNames) 
	{
		cv::Mat image = cv::imread(fileName, cv::IMREAD_COLOR); // Read the image
		if (image.empty()) 
		{
			std::cerr << "Error: Could not load image " << fileName << std::endl;
			continue; // Skip to the next file if loading fails
		}

		imgSize = image.size(); // Get the size of the image
		std::vector<cv::Point2f> keyPointCoordinates = computeCircleCenterCoordinates(image); // Detect keypoints
		size_t isolatedKeypointCount = 5; // Number of isolated keypoints to find
		size_t numNeighbors = 4; // Number of neighbors of isolated keypoints to consider
		
		// Find the most isolated keypoints and its 4 neighbors, total points  = isolatedKeypointCount * (1+ numNeighbors)
		auto selectedKeypointCoordinates = findMostIsolatedPointGroupCoordinates(keyPointCoordinates, isolatedKeypointCount, numNeighbors); 

		//sort the keypoints based on their coordinates
		size_t W = image.size().width;
		std::sort(selectedKeypointCoordinates.begin(), selectedKeypointCoordinates.end(), [&W](const cv::Point2f& a, const cv::Point2f& b) {
			return a.x + W * a.y < b.x + W * b.y;
			});
		imagePoints.push_back(selectedKeypointCoordinates);
		objectPoints.push_back(pattern3D);

		// Display the image with the furthest keypoints marked
		cv::Mat outputImage = image.clone();
		for (const auto& coordinate: selectedKeypointCoordinates)
		{
			cv::circle(outputImage, coordinate, 5, cv::Scalar(0, 255, 0), -1); // Draw a circle around each keypoint
		}
		cv::imshow("Furthest Keypoints", outputImage); // Show the image with keypoints
		cv::waitKey(0); // Wait for a key press to close the window
		cv::destroyAllWindows(); // Close all OpenCV windows
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

std::vector<cv::Point2f> findKNearestNeighborCoords(const cv::Point2f& queryPoint, const std::vector<cv::Point2f>& points, int K)
{
	struct Neighbor
	{
		int index;
		float distance;
	};

	std::vector<Neighbor> neighbors;
	neighbors.reserve(points.size());

	for (int i = 0; i < static_cast<int>(points.size()); i++)
	{
		float dist = cv::norm(queryPoint - points[i]);
		neighbors.push_back({ i, dist });
	}

	std::nth_element(neighbors.begin(), neighbors.begin() + K, neighbors.end(),
		[](const Neighbor& a, const Neighbor& b)
		{
			return a.distance < b.distance;
		});

	neighbors.resize(K);

	std::vector<cv::Point2f> neighborCoords;
	for (const auto& neighbor : neighbors)
	{
		neighborCoords.push_back(points[neighbor.index]);
	}

	return neighborCoords;
}

std::vector<cv::Point2f> findMostIsolatedPointGroupCoordinates(const std::vector<cv::Point2f>& keyPointCoordinates, size_t K, size_t numNeighbors)
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
		std::vector<cv::Point2f> groupNeighborCoords = findKNearestNeighborCoords(keyPointCoordinates[id], keyPointCoordinates, numNeighbors + 1);
		outputKeyPointCoords.insert(outputKeyPointCoords.end(),  groupNeighborCoords.begin(), groupNeighborCoords.end());
	}

	return outputKeyPointCoords; // Return the K most isolated keypoints
}