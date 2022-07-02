from socket import *


server_port = 12001
server_socket = socket(AF_INET, SOCK_STREAM)

server_socket.bind(('0.0.0.0', server_port))  # welcoming socket
server_socket.listen(1)  # listen for TCP connection requests
print("The server is ready to receive")

while True:
    connection_socket, addr = server_socket.accept()  # create a new socket
    message = connection_socket.recv(2048).decode()
    modified_message = message.upper()
    connection_socket.send(modified_message.encode())
    connection_socket.close()

