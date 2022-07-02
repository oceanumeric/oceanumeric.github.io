from socket import *


server_name = '47.108.238.80'
server_port = 12001

client_socket = socket(AF_INET, SOCK_STREAM)  # a tcp connection 

# initiate the TCP connection between client and server 
client_socket.connect((server_name, server_port))

message = input("Please type in lower case: \n")

client_socket.send(message.encode())

modified_message = client_socket.recv(2048)

client_socket.close()