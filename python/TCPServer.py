#!/usr/bin/python

from socket import *
serverPort = 12000

serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind(('',serverPort))
serverSocket.listen(1)

print("The server is ready to recieve")

while True:
	connectionSocket,addr = serverSocket.accept()
	print("Accept new connection from %s:%s..." % addr)
	sentence = connectionSocket.recv(1024)
	print(sentence)
	capitalizedSentence = sentence.upper()
	connectionSocket.send(capitalizedSentence)
	connectionSocket.close()
