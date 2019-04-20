/*----------------------------------------------------------------------------*/
/* Copyright (c) 2018 FIRST. All Rights Reserved.                             */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <iostream>
#include <fstream>

#include "WPIReaderFunctions.h"
#include "UDPHandler.h"
#include "Contour.h"

bool verbose{true};

void flashCameras(int processingVideoSource, int viewingVideoSource)
{
  char buffer[500];

  //Flashes the processingCamera with optimal settings for identifying the targets
  sprintf(buffer,
          "v4l2-ctl -d /dev/video%d \
		--set-ctrl brightness=100 \
		--set-ctrl contrast=255 \
		--set-ctrl saturation=100 \
		--set-ctrl white_balance_temperature_auto=0 \
		--set-ctrl white_balance_temperature=0 \
		--set-ctrl sharpness=24 \
		--set-ctrl gain=24 \
		--set-ctrl exposure_auto=1 \
		--set-ctrl exposure_absolute=100",
          processingVideoSource);
  system(buffer);

  //Makes sure the viewingCamera is set to its optimal settings for actually seeing what's going on
  sprintf(buffer,
          "v4l2-ctl -d /dev/video%d \
		--set-ctrl brightness=0 \
		--set-ctrl contrast=32 \
		--set-ctrl saturation=64 \
		--set-ctrl white_balance_temperature_auto=1 \
		--set-ctrl sharpness=24 \
		--set-ctrl gain=24 \
		--set-ctrl power_line_frequency=2 \
		--set-ctrl exposure_auto=3",
          viewingVideoSource);
  system(buffer);

  if (verbose)
  {
    std::cout << "--- Flashed Cameras ---\n";
  }
}

class MyPipeline : public frc::VisionPipeline
{
public:
  double distanceTo{0},
      verticalAngleError{0},
      horizontalAngleError{0};

  cv::Scalar hsvLow{63, 0, 120},
      hsvHigh{120, 255, 255};

  int minArea{60},
      minRotation{30};

  double horizontalFOV{30},
      verticalFOV{60};

  int width{320}, height{240};

  std::string udpHost{"10.28.51.2"};
  int udpSendPort{1182}, udpReceivePort{1183};
  UDPHandler udpHandler{udpHost, udpSendPort, udpReceivePort};

  cv::Mat morphElement{cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3))};

  bool processingVision{true};
  bool streamVision{true};

  cs::CvSource processingOutputStream = frc::CameraServer::GetInstance()->PutVideo("Processing Camera", width, height);

  int frameCounter{0};

  void extractContours(std::vector<std::vector<cv::Point>> &contours, cv::Mat frame, cv::Scalar &hsvLowThreshold, cv::Scalar &hsvHighThreshold, cv::Mat morphElement)
  {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2HSV);

    //Singles out the pixels that meet the HSV range of the target and displays them
    cv::inRange(frame, hsvLowThreshold, hsvHighThreshold, frame);

    //Applies an open morph to the frame (erosion (dark spaces expand) followed by a dilation (light spaces expand) to remove small particles with a kernel specified by morphElement) and displays it
    //cv::morphologyEx(frame, frame, cv::MORPH_OPEN, morphElement);
    //if (displaySteps)
    //cv::imshow(name + "'s Morph", frame);

    //Shaves down the bright parts of the image and then expands them to remove small false positives
    cv::erode(frame, frame, morphElement, cv::Point(-1, -1), 2);
    cv::dilate(frame, frame, morphElement, cv::Point(-1, -1), 2);

    //Applies the Canny edge detection algorithm to extract edges
    cv::Canny(frame, frame, 0, 0);

    //Finds the contours in the image and stores them in a vector of vectors of cv::Points (each vector of cv::Points represents the curve of the contour)
    //CV_RETR_EXTERNAL specifies to only detect contours on the edges of particles
    //CV_CHAIN_APPROX_SIMPLE compresses the points of the contour to only include their end points
    cv::findContours(frame, contours, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
  }

  void Process(cv::Mat &frame) override
  {
    if (udpHandler.getMessage() == "GO")
    {
      processingVision = true;
    }
    else if (udpHandler.getMessage() == "NO")
    {
      processingVision = false;
    }

    if (frame.empty())
    {
      if (verbose)
      {
        std::cout << "*** Could not open processing frame, retrying... ***\n";
      }
      return;
    }

    ++frameCounter;
    if (frameCounter == 45)
    {
      flashCameras(processingVideoSource, viewingVideoSource);
    }

    if (!processingVision)
    {
      return;
    }

    if (frame.cols != width || frame.rows != height)
      cv::resize(frame, frame, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);

    if (streamVision)
    {
      processingOutputStream.PutFrame(frame);
    }

    std::vector<std::vector<cv::Point>> contoursRaw;
    extractContours(contoursRaw, frame, hsvLow, hsvHigh, morphElement);
    std::vector<Contour> contours(contoursRaw.size());
    for (int i{0}; i < contoursRaw.size(); ++i)
    {
      contours.at(i) = Contour(contoursRaw.at(i));
    }

    if (contours.size() < 2)
    {
      return;
    }

    //Filters out bad contours and adds the contour to the vector
    for (int c{0}; c < contours.size(); ++c)
    {
      if (!contours.at(c).isValid(minArea, minRotation, 3))
      {
        contours.erase(contours.begin() + contours.size() - 1);
        --c;
        continue;
      }
    }

    if (verbose)
    {
      std::cout << "--- Filtered out bad contours ---\n";
    }

    std::vector<std::array<Contour, 2>> pairs{};

    //Least distant contour initialized with -1 so it's not confused for an actual contour and can be tested for not being valid
    int leastDistantContour{-1};

    //Now that we've identified compliant targets, we find their match (if they have one)
    for (int origContour{0}; origContour < contours.size(); ++origContour)
    {
      //We identify the left one first because why not
      if (contours.at(origContour).angle > 0)
      {
        //Iterates through all of the contours and compares them against the original
        for (int compareContour{0}; compareContour < contours.size(); ++compareContour)
        {
          //If the contour to compare against isn't the original
          //and the contour is angled left
          //and the contour is right of the original
          //and (if the least distant contour hasn't been set
          //OR this contour is closer than the last least distant contour)
          //then this contour is the new least distant contour
          if (compareContour != origContour && contours.at(compareContour).angle < 0 && contours.at(origContour).rotatedBoundingBoxPoints[0].x < contours.at(compareContour).rotatedBoundingBoxPoints[0].x)
          {
            //We test if it's closer to the original contour after checking if the
            //index is negative since passing a negative number to a vector will
            //throw an OutOfBounds exception
            if (leastDistantContour == -1)
            {
              leastDistantContour = compareContour;
            }
            else if (contours.at(compareContour).rotatedBoundingBoxPoints[0].x - contours.at(origContour).rotatedBoundingBoxPoints[0].x < contours.at(leastDistantContour).rotatedBoundingBoxPoints[0].x)
            {
              leastDistantContour = compareContour;
            }
          }
        }

        //If we found the second contour, add the pair to the list
        if (leastDistantContour != -1)
        {
          pairs.push_back(std::array<Contour, 2>{contours.at(origContour), contours.at(leastDistantContour)});
          break;
        }
      }
    }

    if (verbose)
    {
      std::cout << "--- Matched contour pairs ---\n";
    }

    if (pairs.size() == 0)
    {
      return;
    }

    std::array<Contour, 2> closestPair{pairs.back()};
    for (int p{0}; p < pairs.size(); ++p)
    {
      double comparePairCenter{((std::max(pairs.at(p).at(0).rotatedBoundingBox.center.x, pairs.at(p).at(1).rotatedBoundingBox.center.x) - std::min(pairs.at(p).at(0).rotatedBoundingBox.center.x, pairs.at(p).at(1).rotatedBoundingBox.center.x)) / 2) + std::min(pairs.at(p).at(0).rotatedBoundingBox.center.x, pairs.at(p).at(1).rotatedBoundingBox.center.x)};
      double closestPairCenter{((std::max(closestPair.at(0).rotatedBoundingBox.center.x, closestPair.at(1).rotatedBoundingBox.center.x) - std::min(closestPair.at(0).rotatedBoundingBox.center.x, closestPair.at(1).rotatedBoundingBox.center.x)) / 2) + std::min(closestPair.at(0).rotatedBoundingBox.center.x, closestPair.at(1).rotatedBoundingBox.center.x)};

      if (std::abs(comparePairCenter) - (width / 2) <
          std::abs(closestPairCenter) - (width / 2))
      {
        closestPair = std::array<Contour, 2>{pairs.at(p).at(0), pairs.at(p).at(1)};
      }
    }

    if (verbose)
    {
      std::cout << "--- Found pairs closest to the center ---\n";
    }

    //For clarity
    double centerX{((std::max(closestPair.at(0).rotatedBoundingBox.center.x, closestPair.at(1).rotatedBoundingBox.center.x) - std::min(closestPair.at(0).rotatedBoundingBox.center.x, closestPair.at(1).rotatedBoundingBox.center.x)) / 2) + std::min(closestPair.at(0).rotatedBoundingBox.center.x, closestPair.at(1).rotatedBoundingBox.center.x)};
    double centerY{((std::max(closestPair.at(0).rotatedBoundingBox.center.y, closestPair.at(1).rotatedBoundingBox.center.y) - std::min(closestPair.at(0).rotatedBoundingBox.center.y, closestPair.at(1).rotatedBoundingBox.center.y)) / 2) + std::min(closestPair.at(0).rotatedBoundingBox.center.x, closestPair.at(1).rotatedBoundingBox.center.y)};

    //The original contour will always be the left one since that's what we've specified
    //Calculates and spits out some values for us
    //distanceTo = (regression function);
    horizontalAngleError = -((frame.cols / 2.0) - centerX) / frame.cols * horizontalFOV;
    //verticalAngleError = ((processingFrame.rows / 2.0) - centerY) / frame.rows * horizontalFOV;

    //double height = closestPair.at(0).rotatedBoundingBox.size.width;

    /*
		height(pixels) / vertical(total pixels) = 6.31(height of tape in inches) / height(of frame in inches)
		height of frame(inches) = 6.31 * vertical(pixels) / height(pixels)
		
		tan(30) = 0.5*height of frame(inches) / distance
		distance = 0.5*height of frame(inches) / tan(30)
		
		distance = 0.5 * 6.31 * vertical(pixels) / height(pixels) / tan(vertical FOV / 2)
		*/
    //double distance = 0.5 * 6.31 * frame.rows / height / std::tan(verticalFOV * 0.5 * 3.141592654 / 180); //1751.45 / height; //.1945 * height * height + -7.75 * height + 122.4;

    // Conversion to radians (the std trigonometry functions only take radians)
    //horizontalAngleError *= 3.141592654 / 180;

    //horizontalAngleError = std::atan(distance * std::sin(horizontalAngleError) / (distance * std::cos(horizontalAngleError) - 7.5));

    // Conversion back to degrees
    //horizontalAngleError *= 180 / 3.141592654;

    udpHandler.send(std::to_string(horizontalAngleError));

    //std::cout << "Max - min y: " << closestPair.at(0).rotatedBoundingBoxPoints[3] -  closestPair.at(0).rotatedBoundingBoxPoints[1] << "\n\n";

    //std::cout << "Height (in pixels): " << height << '\n';
    //std::cout << "Distance: " << distance << '\n';
    //std::cout << "AOE: " << horizontalAngleError << "\n\n";

    if (verbose)
    {
      std::cout << "--- Sent angle of error to the roboRIO ---\n";
    }
  }
};

int main(int argc, char *argv[])
{
  if (argc >= 2)
    configFile = argv[1];

  if (!ReadConfig())
    return EXIT_FAILURE;

  // The Pi recognizes both cameras correctly more often when this command is issued
  system("ls /dev/video*");

  FILE *uname;
  char consoleOutput[300];
  int lastchar;

  // Executes the command supplied to popen and saves the output in the char array
  uname = popen("v4l2-ctl --list-devices", "r");
  lastchar = fread(consoleOutput, 1, 300, uname);
  consoleOutput[lastchar] = '\0';

  // Converts the char array to an std::string
  std::string outputString = consoleOutput;

  /**
	 * Working from the inside out:
	 * 	- Finds where the camera's name is in the string
	 * 	- Uses the location of the first character in that string as the starting point for a new search for the location of the first character in /dev/video
	 * 	- Looks ten characters down the string to find the number that comes after /dev/video
	 * 	- Parses the output character for an integer
	 * 	- Assigns that integer to the appropriate variable
	 */
  viewingVideoSource = outputString.at(outputString.find("/dev/video", outputString.find("USB 2.0 Camera: HD USB Camera")) + 10) - '0';
  processingVideoSource = outputString.at(outputString.find("/dev/video", outputString.find("UVC Camera")) + 10) - '0';
  pclose(uname);

  // start cameras
  for (const auto &config : cameraConfigs)
  {
    if (config.name == "Fisheye")
    {
      cameras.emplace_back(StartCameraAndStream(config));
    }
    else
    {
      cameras.emplace_back(StartCamera(config));
    }
  }

  // Flashes the cameras with the wrong settings followed by the right settings
  // This is more reliable than just flashing the good settings (I don't know why)
  flashCameras(viewingVideoSource, processingVideoSource);
  flashCameras(processingVideoSource, viewingVideoSource);

  int viewingVideoIndex{cameraConfigs.at(0).name == "Fisheye" ? 0 : 1},
      processingVideoIndex{cameraConfigs.at(0).name == "Fisheye" ? 1 : 0};

  // start image processing on the processing camera if present
  if (cameras.size() >= 1)
  {
    std::thread([&] {
      frc::VisionRunner<MyPipeline> runner(cameras.at(processingVideoIndex), new MyPipeline(),
                                           [&](MyPipeline &pipeline) {
                                             // do something with pipeline results
                                           });
      runner.RunForever();
    })
        .detach();
  }

  char buffer[500];
  sprintf(buffer,
          "v4l2-ctl --device=/dev/video%d --list-ctrls &",
          processingVideoSource);

  // loop forever
  for (;;)
  {
    // Re-flashes the camera every fifteen seconds to ensure that they were calibrated correctly
    flashCameras(processingVideoSource, viewingVideoSource);

    system(buffer);

    std::this_thread::sleep_for(std::chrono::seconds(15));
  }
}
