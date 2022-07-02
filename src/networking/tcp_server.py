from socket import *

server_host = 'localhost'  # receive all interface
server_port = 12000

with socket(AF_INET, SOCK_STREAM) as stpc:
    stpc.bind((server_host, server_port))  # welcoming socket
    stpc.listen(1)  # listen for TCP connection requests
    print("The server is ready to receive")
    while True:
        connection_socket, addr = stpc.accept()  # create a new socket
        print(f"Connected by {addr}")
        message = connection_socket.recv(1024).decode()
        print(message)
        modified_message = message.upper()
        connection_socket.send(modified_message.encode())
        connection_socket.close()
        
            
    

