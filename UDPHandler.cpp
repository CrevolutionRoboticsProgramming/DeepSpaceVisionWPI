#include "UDPHandler.h"

UDPHandler::UDPHandler(std::string ip, int sendPort, int receivePort)
	: ip{ip}, sendPort{sendPort}
{
	socket.open(ip::udp::v4(), error);
	send_endpoint = ip::udp::endpoint(ip::address::from_string(ip), sendPort);
	receive_endpoint = ip::udp::endpoint(ip::udp::v4(), receivePort);
	socket.bind(receive_endpoint, error);
	service.run(error);

	receiveThread = std::thread(&UDPHandler::receive, this);
}

void UDPHandler::send(std::string message)
{
	socket.send_to(buffer(message, bufferSize), send_endpoint, 0, error);
}

void UDPHandler::receive()
{
	while (true)
	{
		char buf[bufferSize];
		socket.receive(buffer(buf, bufferSize), 0, error);
		receivedMessage = std::string(buf);
	}
}

void UDPHandler::close()
{
	socket.close();
}

std::string UDPHandler::getMessage()
{
	return std::string(receivedMessage);
}

void UDPHandler::clearMessage()
{
	receivedMessage.clear();
}

std::string UDPHandler::getIP()
{
	return ip;
}

int UDPHandler::getSendPort()
{
	return sendPort;
}

int UDPHandler::getReceivePort()
{
	return receivePort;
}
