#pragma once

#include <boost/asio.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <thread>

using namespace boost::asio;

class UDPHandler {
private:
	std::string ip;
	int sendPort, receivePort;

	io_service service;
	ip::udp::socket socket{ service };
	ip::udp::endpoint receive_endpoint, send_endpoint;
	boost::system::error_code error;

	static const int bufferSize{ 8 };

	std::string receivedMessage;
	std::thread receiveThread;

	void receive();

public:
	UDPHandler(std::string ip, int sendPort, int receivePort);
	void send(std::string message);
	void close();
	std::string getMessage();
	void clearMessage();
	std::string getIP();
	int getSendPort();
	int getReceivePort();
};
