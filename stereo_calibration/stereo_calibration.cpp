#include <iostream>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <algorithm>

using std::uint64_t;


static uint64_t Solution(uint64_t A);
void calibrateStereoCamera(const cv::Mat& leftImage, const cv::Mat& rightImage,const std::vector<cv::Point3f>& pattern3D, double& rms,	cv::Mat& rightCameraPosition, double& camerAngularDeviation);
std::vector<cv::Point2f> computeCircleCenterCoordinates(const cv::Mat& image);
std::vector<cv::Point2f> findKNearestNeighborCoords(const cv::Point2f& queryPoint, const std::vector<cv::Point2f>& points, int K);
std::vector<cv::Point2f> findMostIsolatedPointGroupCoordinates(const std::vector<cv::Point2f>& keyPointCoordinates, size_t K = 5, size_t numNeighbors = 4);
std::vector<cv::Point3f> generate3DCoordinates(const std::vector<cv::Point3f>& basePoints, const std::vector<cv::Point3f>& offsets);
cv::Mat cropByKeypoints(const cv::Mat& img, const std::vector<cv::Point2f>& keypoints, size_t padding);


int main()
{
	Solution(0);
	return 0;
}
//========//========//========//========//=======#//========//========//========//========//=======#

uint64_t Solution(uint64_t A)
{
	double baseUnit = 1.0; // mm
	// patter3d setting
	std::vector<cv::Point3f> pattern3D = {};
	{
		std::vector<cv::Point3f> basePoints
		{
			cv::Point3f(-baseUnit , baseUnit , 0),
			cv::Point3f(baseUnit , baseUnit , 0),
			cv::Point3f(0, 0, 0),
			cv::Point3f(-baseUnit , -baseUnit , 0),
			cv::Point3f(baseUnit , -baseUnit , 0)
		};

		std::vector<cv::Point3f> offsets
		{
			cv::Point3f(0, 0, 0),
			cv::Point3f(-4 * baseUnit, 0, 0),
			cv::Point3f(4 * baseUnit, 0, 0),
			cv::Point3f(0, 4 * baseUnit, 0),
			cv::Point3f(0, -4 * baseUnit, 0)
		};

		pattern3D = generate3DCoordinates(basePoints, offsets);
	}

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
			images.push_back(cv::Mat());
		}
		images.push_back(image);
	}
	
	if (images.size() < 2) 
	{
		std::cerr << "Error: Not enough images for stereo calibration." << std::endl;
		return 1; // Exit if not enough images
	}

	// 3. Calibrate stereo camera
	for (size_t i = 1; i < images.size(); i++) 
	{
		cv::Mat leftImage = images[i - 1];
		cv::Mat rightImage = images[i];

		double rms, camerAngularDeviation;
		cv::Mat rightCameraPosition;
		calibrateStereoCamera(leftImage, rightImage, pattern3D, rms, rightCameraPosition, camerAngularDeviation);
		std::cout << "Stereo calibration RMS error: " << rms << std::endl;
		std::cout << "Right camera position: " << rightCameraPosition.t() << std::endl; // Transpose for better readability
		std::cout << "Baseline: " << baseUnit << "mm" << std::endl;
		std::cout << "camera deviation angle: " << camerAngularDeviation << " degrees" << std::endl;
	}

	return 0;
}

void calibrateStereoCamera(const cv::Mat& leftImage, const cv::Mat& rightImage, const std::vector<cv::Point3f>& pattern3D, double& rms, cv::Mat& rightCameraPosition, double& camerAngularDeviation)
{
	if (leftImage.size() != rightImage.size())
	{
		std::cerr << "Error: Images must be of the same size for stereo calibration." << std::endl;
		return;
	}

	std::vector<std::vector<cv::Point2f>> imagePoints; // Store 2D points from images
	std::vector<std::vector<cv::Point3f>> objectPoints;
	cv::Size imgSize = leftImage.size();
	std::vector<cv::Mat> croppedImages = {};

	for (const auto& image : { leftImage, rightImage })
	{
		std::vector<cv::Point2f> keyPointCoordinates = computeCircleCenterCoordinates(image); // Detect keypoints
		size_t isolatedKeypointCount = 5; // Number of isolated keypoints to find
		size_t numNeighbors = 4; // Number of neighbors of isolated keypoints to consider

		// Find the most isolated keypoints and its 4 neighbors, total points  = isolatedKeypointCount * (1+ numNeighbors)
		auto selectedKeypointCoordinates = findMostIsolatedPointGroupCoordinates(keyPointCoordinates, isolatedKeypointCount, numNeighbors);

		//sort the keypoints based on their coordinates
		std::sort(selectedKeypointCoordinates.begin(), selectedKeypointCoordinates.end(), [&imgSize](const cv::Point2f& a, const cv::Point2f& b) {
			return a.x + imgSize.width * a.y < b.x + imgSize.width * b.y;
			});
		imagePoints.push_back(selectedKeypointCoordinates);
		objectPoints.push_back(pattern3D);

		// Display the image with the furthest keypoints marked
		cv::Mat outputImage = image.clone();
		for (const auto& coordinate : selectedKeypointCoordinates)
		{
			//std::cout << "coord: " << coordinate.x << "," << coordinate.y << std::endl;
			cv::circle(outputImage, coordinate, 5, cv::Scalar(0, 255, 0), -1); // Draw a circle around each keypoint
		}

		croppedImages.push_back(cropByKeypoints(image, selectedKeypointCoordinates, 5));
	}
	cv::Mat cameraMatrix, distCoeffs;
	std::vector<cv::Mat> rvecs, tvecs;

	rms = calibrateCamera(
		objectPoints,
		imagePoints,
		imgSize,
		cameraMatrix,
		distCoeffs,
		rvecs,
		tvecs
	);

	for (const auto& image : croppedImages)
	{
		cv::imshow("Cropped Image", image); // Show the undistorted image
		cv::waitKey(0); // Wait for a key press to close the window
	}

	auto cropLeftImage = croppedImages[0];
	auto cropRightImage = croppedImages[1];
	cv::cvtColor(cropLeftImage, cropLeftImage, cv::COLOR_BGR2GRAY);
	cv::cvtColor(cropRightImage, cropRightImage, cv::COLOR_BGR2GRAY);

	cv::resize(cropLeftImage, cropLeftImage, cv::Size(400,400));
	cv::resize(cropRightImage, cropRightImage, cv::Size(400,400));

	cv::threshold(cropLeftImage, cropLeftImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	cv::threshold(cropRightImage, cropRightImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	imshow("Left Image", cropLeftImage); // Show the left image
	imshow("Right Image", cropRightImage); // Show the right image
	cv::waitKey(0); // Wait for a key press to close the window

	cv::Mat xor_img, rot_image = cropLeftImage;
	std::vector<size_t> nonzero_count;
	for (size_t i = 0; i < 4; i++)
	{
		cv::bitwise_xor(rot_image, cropRightImage, xor_img);
		nonzero_count.push_back(cv::countNonZero(xor_img));// XOR operation to visualize differences
		cv::rotate(rot_image, rot_image, cv::ROTATE_90_CLOCKWISE);
		std::cout << nonzero_count.back() << std::endl;
	}
	cv::bitwise_xor(cropLeftImage, cropRightImage, cropLeftImage); // XOR operation to visualize differences

	cv::destroyAllWindows(); // Close all OpenCV windows

	cv::Mat Rl, Rr;
	cv::Rodrigues(rvecs[0], Rl); // (rotation axis * rotation angle) vector to rotation matrix
	cv::Rodrigues(rvecs[1], Rr); // (rotation axis * rotation angle) vector to rotation matrix

	cv::Mat tl = tvecs[0];
	cv::Mat tr = tvecs[1];

	// inverse map
	cv::Mat Rl_inv = Rl.t(); // Rl is rotation matrix which is orthogonal, thus Rl_inv = Rl^T

	// xl = Rl * x + tl
	// xr = Rr * x + tr
	// xr = Rr * (Rl_inv * (xl - tl)) + tr
	// xr = Rr * Rl_inv * xl - Rr * Rl_inv * tl + tr

	rightCameraPosition = -Rr * Rl_inv * tl + tr;
	cv::Mat rotVec;
	cv::Rodrigues(Rr * Rl_inv, rotVec);
	camerAngularDeviation = cv::norm(rotVec) * 180.0 / CV_PI; // degree
}

std::vector<cv::Point3f> generate3DCoordinates(const std::vector<cv::Point3f>& basePoints, const std::vector<cv::Point3f>& offsets)
{
	// This function is a placeholder for computing 3D coordinates.
	// In a real application, you would implement the logic to compute 3D coordinates based on stereo calibration.

	std::vector<cv::Point3f> combinedPoints;

	for (const auto& b : basePoints)
	{
		for (const auto& o : offsets)
		{
			combinedPoints.push_back(cv::Point3f(b.x + o.x, b.y + o.y, 0));
		}
	}

	std::sort(combinedPoints.begin(), combinedPoints.end(), [](const cv::Point3f& p1, const cv::Point3f& p2) {
			if (p1.y!= p2.y)
			{
				return p1.y > p2.y;  // y descending order
			}
			else
			{
				return p1.x < p2.x;  // x ascending order
			}
		});

	//std::cout << "Sorted points (x asc, y desc): ";
	//for (const auto& p : combinedPoints)
	//{
	//	std::cout << "(" << p.x << ", " << p.y << ", " << p.z << ")\n";
	//}
	return combinedPoints;
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
			if (neighborCoord == keypointCoord) 
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

	std::cout << "Detected circles: " << outputKeyPointCoords.size() << std::endl;

	std::ostringstream oss;
	oss << "Circle center coordinates: ";
	for (const auto& keyPointCoord : outputKeyPointCoords)
	{
		oss << "(" << keyPointCoord.x << "," << keyPointCoord.y << "), ";
	}
	auto coordString = oss.str();
	coordString.resize(coordString.size() - 2); // remove last comma
	std::cout << coordString << std::endl;

	return outputKeyPointCoords; // Return the K most isolated keypoints
}

cv::Mat cropByKeypoints(const cv::Mat& img, const std::vector<cv::Point2f>& keypoints, size_t padding)
{
	int x_min = keypoints[0].x;
	int x_max = keypoints[0].x;
	int y_min = keypoints[0].y;
	int y_max = keypoints[0].y;
	auto imgSize = img.size();

	for (const auto& pt : keypoints) 
	{
		x_min = std::min(x_min, int(pt.x - padding));
		y_min = std::min(y_min, int(pt.y - padding));

		x_max = std::max(x_max, int(pt.x + padding));
		y_max = std::max(y_max, int(pt.y + padding));
	}

	x_min = std::max(0, x_min);
	y_min = std::max(0, y_min);
	x_max = std::min(imgSize.width, x_max);
	y_max = std::min(imgSize.height, y_max);
	cv::Rect roi(x_min, y_min, x_max - x_min, y_max - y_min);

	return img(roi).clone();
}
