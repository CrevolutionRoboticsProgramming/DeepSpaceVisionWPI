#pragma once

#include <string>
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

unsigned int team;
bool server = false;
static const char *configFile = "/boot/frc.json";

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

int processingVideoSource{}, viewingVideoSource{};

int width{320}, height{240};

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
  cs::UsbCamera camera{config.name, config.path};

  camera.SetConfigJson(config.config);

  return camera;
}

cs::UsbCamera StartCameraAndStream(const CameraConfig &config)
{
  cs::UsbCamera camera{StartCamera(config)};

  auto inst = frc::CameraServer::GetInstance();
  auto server = inst->StartAutomaticCapture(camera);

  camera.SetConnectionStrategy(cs::VideoSource::kConnectionKeepOpen);

  if (config.streamConfig.is_object())
    server.SetConfigJson(config.streamConfig);

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