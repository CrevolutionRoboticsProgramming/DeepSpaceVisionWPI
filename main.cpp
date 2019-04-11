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

#include <networktables/NetworkTableInstance.h>
#include <vision/VisionPipeline.h>
#include <vision/VisionRunner.h>
#include <wpi/StringRef.h>
#include <wpi/json.h>
#include <wpi/raw_istream.h>
#include <wpi/raw_ostream.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "cameraserver/CameraServer.h"
#include "UDPHandler.h"
#include "Contour.h"

/*
   JSON format:
   {
       "team": <team number>,
       "ntmode": <"client" or "server", "client" if unspecified>
       "cameras": [
           {
               "name": <camera name>
               "path": <path, e.g. "/dev/video0">
               "pixel format": <"MJPEG", "YUYV", etc>   // optional
               "width": <video mode width>              // optional
               "height": <video mode height>            // optional
               "fps": <video mode fps>                  // optional
               "brightness": <percentage brightness>    // optional
               "white balance": <"auto", "hold", value> // optional
               "exposure": <"auto", "hold", value>      // optional
               "properties": [                          // optional
                   {
                       "name": <property name>
                       "value": <property value>
                   }
               ],
               "stream": {                              // optional
                   "properties": [
                       {
                           "name": <stream property name>
                           "value": <stream property value>
                       }
                   ]
               }
           }
       ]
       "switched cameras": [
           {
               "name": <virtual camera name>
               "key": <network table key used for selection>
               // if NT value is a string, it's treated as a name
               // if NT value is a double, it's treated as an integer index
           }
       ]
   }
 */

static const char *configFile = "/boot/frc.json";

namespace
{

unsigned int team;
bool server = false;

struct CameraConfig
{
  std::string name;
  std::string path;
  wpi::json config;
  wpi::json streamConfig;
};

struct SwitchedCameraConfig
{
  std::string name;
  std::string key;
};

std::vector<CameraConfig> cameraConfigs;
std::vector<SwitchedCameraConfig> switchedCameraConfigs;
std::vector<cs::VideoSource> cameras;

int videoSource{};

IplImage *mapx, *mapy;

int width{160}, height{120};

wpi::raw_ostream &ParseError()
{
  return wpi::errs() << "config error in '" << configFile << "': ";
}

bool ReadCameraConfig(const wpi::json &config)
{
  CameraConfig c;

  // name
  try
  {
    c.name = config.at("name").get<std::string>();
  }
  catch (const wpi::json::exception &e)
  {
    ParseError() << "could not read camera name: " << e.what() << '\n';
    return false;
  }

  // path
  try
  {
    c.path = config.at("path").get<std::string>();
  }
  catch (const wpi::json::exception &e)
  {
    ParseError() << "camera '" << c.name
                 << "': could not read path: " << e.what() << '\n';
    return false;
  }

  // stream properties
  if (config.count("stream") != 0)
    c.streamConfig = config.at("stream");

  c.config = config;

  cameraConfigs.emplace_back(std::move(c));
  return true;
}

bool ReadSwitchedCameraConfig(const wpi::json &config)
{
  SwitchedCameraConfig c;

  // name
  try
  {
    c.name = config.at("name").get<std::string>();
  }
  catch (const wpi::json::exception &e)
  {
    ParseError() << "could not read switched camera name: " << e.what() << '\n';
    return false;
  }

  // key
  try
  {
    c.key = config.at("key").get<std::string>();
  }
  catch (const wpi::json::exception &e)
  {
    ParseError() << "switched camera '" << c.name
                 << "': could not read key: " << e.what() << '\n';
    return false;
  }

  switchedCameraConfigs.emplace_back(std::move(c));
  return true;
}

bool ReadConfig()
{
  // open config file
  std::error_code ec;
  wpi::raw_fd_istream is(configFile, ec);
  if (ec)
  {
    wpi::errs() << "could not open '" << configFile << "': " << ec.message()
                << '\n';
    return false;
  }

  // parse file
  wpi::json j;
  try
  {
    j = wpi::json::parse(is);
  }
  catch (const wpi::json::parse_error &e)
  {
    ParseError() << "byte " << e.byte << ": " << e.what() << '\n';
    return false;
  }

  // top level must be an object
  if (!j.is_object())
  {
    ParseError() << "must be JSON object\n";
    return false;
  }

  // team number
  try
  {
    team = j.at("team").get<unsigned int>();
  }
  catch (const wpi::json::exception &e)
  {
    ParseError() << "could not read team number: " << e.what() << '\n';
    return false;
  }

  // ntmode (optional)
  if (j.count("ntmode") != 0)
  {
    try
    {
      auto str = j.at("ntmode").get<std::string>();
      wpi::StringRef s(str);
      if (s.equals_lower("client"))
      {
        server = false;
      }
      else if (s.equals_lower("server"))
      {
        server = true;
      }
      else
      {
        ParseError() << "could not understand ntmode value '" << str << "'\n";
      }
    }
    catch (const wpi::json::exception &e)
    {
      ParseError() << "could not read ntmode: " << e.what() << '\n';
    }
  }

  // cameras
  try
  {
    for (auto &&camera : j.at("cameras"))
    {
      if (!ReadCameraConfig(camera))
        return false;
    }
  }
  catch (const wpi::json::exception &e)
  {
    ParseError() << "could not read cameras: " << e.what() << '\n';
    return false;
  }

  // switched cameras (optional)
  if (j.count("switched cameras") != 0)
  {
    try
    {
      for (auto &&camera : j.at("switched cameras"))
      {
        if (!ReadSwitchedCameraConfig(camera))
          return false;
      }
    }
    catch (const wpi::json::exception &e)
    {
      ParseError() << "could not read switched cameras: " << e.what() << '\n';
      return false;
    }
  }

  return true;
}

cs::UsbCamera StartCamera(const CameraConfig &config)
{
  wpi::outs() << "Starting camera '" << config.name << "' on " << config.path
              << '\n';
  auto inst = frc::CameraServer::GetInstance();
  cs::UsbCamera camera{config.name, config.path};
  //auto server = inst->StartAutomaticCapture(camera);

  camera.SetConfigJson(config.config);
  camera.SetConnectionStrategy(cs::VideoSource::kConnectionKeepOpen);

  //if (config.streamConfig.is_object())
  //  server.SetConfigJson(config.streamConfig);

  return camera;
}

cs::MjpegServer StartSwitchedCamera(const SwitchedCameraConfig &config)
{
  wpi::outs() << "Starting switched camera '" << config.name << "' on "
              << config.key << '\n';
  auto server =
      frc::CameraServer::GetInstance()->AddSwitchedCamera(config.name);

  nt::NetworkTableInstance::GetDefault()
      .GetEntry(config.key)
      .AddListener(
          [server](const auto &event) mutable {
            if (event.value->IsDouble())
            {
              int i = event.value->GetDouble();
              if (i >= 0 && i < cameras.size())
                server.SetSource(cameras[i]);
            }
            else if (event.value->IsString())
            {
              auto str = event.value->GetString();
              for (int i = 0; i < cameraConfigs.size(); ++i)
              {
                if (str == cameraConfigs[i].name)
                {
                  server.SetSource(cameras[i]);
                  break;
                }
              }
            }
          },
          NT_NOTIFY_IMMEDIATE | NT_NOTIFY_NEW | NT_NOTIFY_UPDATE);

  return server;
}

void flashCamera()
{
  char buffer[500];
  //Makes sure the viewingCamera is set to its optimal settings for actually seeing what's going on
  sprintf(buffer,
          "v4l2-ctl -d /dev/video%d \
		--set-ctrl brightness=0 \
		--set-ctrl contrast=32 \
		--set-ctrl saturation=60 \
		--set-ctrl white_balance_temperature_auto=0 \
		--set-ctrl white_balance_temperature=0 \
		--set-ctrl sharpness=2 \
		--set-ctrl gain=0 \
		--set-ctrl exposure_auto=1 \
		--set-ctrl exposure_absolute=105",
          videoSource);
  system(buffer);
}

class MyPipeline : public frc::VisionPipeline
{
public:
  bool verbose{true};

  double distanceTo{0},
      verticalAngleError{0},
      horizontalAngleError{0};

  cv::Scalar hsvLow{70, 150, 230},
      hsvHigh{110, 210, 255};

  int minArea{60},
      minRotation{30};

  double horizontalFOV{30},
      verticalFOV{60};

  int width{160}, height{120};

  std::string udpHost{"10.28.51.2"};
  int udpSendPort{1182}, udpReceivePort{1183};
  UDPHandler udpHandler{udpHost, udpSendPort, udpReceivePort};

  cv::Mat morphElement{cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3))};

  bool processingVision{false};

  cs::CvSource processingOutputStream = frc::CameraServer::GetInstance()->PutVideo("Processing Camera", 160, 120);

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
        std::cout << "*** Could not open video transmitting frame, retrying... ***\n";
      }
      return;
    }

    if (frameCounter == 45)
    {
      flashCamera();
      ++frameCounter;
    }
    else
    {
      ++frameCounter;
    }

    cv::line(frame, cv::Point(frame.cols / 2, 0), cv::Point(frame.cols / 2, frame.rows), cv::Scalar(0, 0, 0), 1.5);

    processingOutputStream.PutFrame(frame);

    if (!processingVision)
    {
      return;
    }

    IplImage img = frame;

    cvRemap(&img, &img, mapx, mapy);

    frame = cv::cvarrToMat(&img);

    std::vector<std::vector<cv::Point>> contoursRaw;
    extractContours(contoursRaw, frame, hsvLow, hsvHigh, morphElement);
    std::vector<Contour> contours(contoursRaw.size());
    for (int i{0}; i < contoursRaw.size(); ++i)
    {
      contours.at(i) = Contour(contoursRaw.at(i));
    }

    if (verbose)
    {
      std::cout << "--- Extracted contours ---\n";
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
} // namespace

int main(int argc, char *argv[])
{
  if (argc >= 2)
    configFile = argv[1];

  // read configuration
  if (!ReadConfig())
    return EXIT_FAILURE;

  // start NetworkTables
  auto ntinst = nt::NetworkTableInstance::GetDefault();
  if (server)
  {
    wpi::outs() << "Setting up NetworkTables server\n";
    ntinst.StartServer();
  }
  else
  {
    wpi::outs() << "Setting up NetworkTables client for team " << team << '\n';
    ntinst.StartClientTeam(team);
  }

  std::cout << "Beginning\n";

  CvMat *intrinsic{(CvMat *)cvLoad("Intrinsics.xml")};
  CvMat *distortion{(CvMat *)cvLoad("Distortion.xml")};

  std::cout << "Second\n";

  mapx = cvCreateImage(cv::Size(width, height), IPL_DEPTH_32F, 1);
  mapy = cvCreateImage(cv::Size(width, height), IPL_DEPTH_32F, 1);

  std::cout << "Fourth\n";

  cvInitUndistortMap(intrinsic, distortion, mapx, mapy);

  std::cout << "Sixth\n";

  system("ls /dev/video*");

  FILE *uname;
  char consoleOutput[300];
  int lastchar;

  // Executes the command supplied to popen and saves the output in the char array
  uname = popen("v4l2-ctl --list-devices", "r");
  lastchar = fread(consoleOutput, 1, 300, uname);
  consoleOutput[lastchar] = '\0';

  // Converts the char array to a std::string
  std::string outputString = consoleOutput;

  /**
	 * Working from the inside out:
	 * 	- Finds where the camera's name is in the string
	 * 	- Uses the location of the first character in that string as the starting point for a new search for the location of the first character in /dev/video
	 * 	- Looks ten characters down the string to find the number that comes after /dev/video
	 * 	- Parses the output character for an integer
	 * 	- Assigns that integer to the appropriate variable
	 */
  videoSource = outputString.at(outputString.find("/dev/video", outputString.find("USB 2.0 Camera: HD USB Camera")) + 10) - '0';
  pclose(uname);

  // start cameras
  for (const auto &config : cameraConfigs)
    cameras.emplace_back(StartCamera(config));

  flashCamera();

  // start image processing on the processing camera if present
  if (cameras.size() >= 1)
  {
    std::thread([&] {
      frc::VisionRunner<MyPipeline> runner(cameras[0], new MyPipeline(),
                                           [&](MyPipeline &pipeline) {
                                             // do something with pipeline results
                                           });
      runner.RunForever();
    })
        .detach();
  }

  // loop forever
  for (;;)
    std::this_thread::sleep_for(std::chrono::seconds(10));
}
