from socket import * 


server_name = '47.108.238.80'
server_port = 12000

client_socket = socket(AF_INET, SOCK_DGRAM)
message = input('Input lowercase sentence:')

client_socket.sendto(message.encode(), (server_name, server_port))

modified_message, server_address = client_socket.recvfrom(2048)

print(modified_message.decode())

client_socket.close()
