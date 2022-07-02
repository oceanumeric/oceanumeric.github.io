from socket import *


server_name = '47.108.238.80'
server_port = 12000

client_socket = socket(AF_INET, SOCK_STREAM)

try:
    client_socket.connect((server_name, server_port))
except:
    print("connection failed")

message = input("What's you message: \n")

client_socket.send(message.encode())

reply_message = client_socket.recv(1024)

print('From Server: ', reply_message.decode())

client_socket.close()