from socket import *


server_name = '47.108.238.80'
server_port = 65432

with socket(AF_INET, SOCK_STREAM) as s:  # a tcp connection 
    # initiate the TCP connection between client and server 
    s.connect((server_name, server_port))
    message = input("Please type in lower case: \n")
    s.send(message.encode())
    modified_message = s.recv(1024)
    print(modified_message.decode())
    s.close()