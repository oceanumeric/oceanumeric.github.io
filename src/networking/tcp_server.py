from socket import *

server_host = '0.0.0.0'
server_port = 65432

with socket(AF_INET, SOCK_STREAM) as stpc:
    stpc.bind((server_host, server_port))  # welcoming socket
    stpc.listen()  # listen for TCP connection requests
    connection_socket, addr = stpc.accept()  # create a new socket
    print("The server is ready to receive")
    with connection_socket:
        print(f"Connected by {addr}")
        while True:
            message = connection_socket.recv(1024).decode()
            modified_message = message.upper()
            connection_socket.send(modified_message.encode())
            connection_socket.close()
    

