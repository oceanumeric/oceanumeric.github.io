# Socket Programming

In this post, we will write simple client-server programs that use user
datagram protocol (UDP) and transmission control protocol (TCP). Recall that
TCP is connection oriented (meaning that the communicating devices should 
establish a connection before transmitting data and should close the connection
after transmitting the data.) and provides a reliable byte-stream channel.
However, UDP is connectionless and sends independent packets of data from
one end system to the other, without any guarantees about deliver. 

We will use the following simple client-server application to demonstrate socket
programming for both UDP and TCP:

1. The client reads a line of characters (data) from its keyboard and sends
the data to the server.
2. The server receives the data and converts the characters to uppercase
3. The server sends the modified data to the client
4. The client receives the modified data and displays the line on its screen 


## Socket programming with UDP

To test our socket programming, we need client and server. I will run the `udp_client.py`
script in my computer and run `udp_sever.py` in an instance I bought from aliyun
(you could buy one from DigitalOcean or GoogleCloud). To make sure the instance
of cloud server, you need to open the port first as following.

![aliyu-port](./images/aliyun.png)

We will login my cloud server and download the `udp_server.py` into a file
called `cs144`, then just run it. The server will start to listen. 

```bash
ssh -p 22 root@47.108.238.80   # my ssh port is 22
wget https://raw.githubusercontent.com/oceanumeric/oceanumeric.github.io/main/src/networking/udp_server.py
python3 udp_server.py  # it should print The server is ready to receive
# when you finish the session, type
exit
```

Then you can run `udp_client.py` on your computer and it will send messages 
to the server and return the strings with upper case. 


=== "udp_client.py"
    ```py
    from socket import * 


    server_name = '47.108.238.80'
    server_port = 12000

    client_socket = socket(AF_INET, SOCK_DGRAM)

    message = input("Please type in lower case: \n")

    client_socket.sendto(message.encode(), (server_name, server_port))

    modified_message, server_address = client_socket.recvfrom(2048)

    print(modified_message.decode())

    client_socket.close()
    ```
=== "udp_server.py"
    ```py
    from socket import * 


    server_port = 12000
    server_socket = socket(AF_INET, SOCK_DGRAM)

    server_socket.bind(('0.0.0.0', server_port))

    print("The server is ready to receive")

    while True:
        message, client_address = server_socket.recvfrom(2048)
        print(message.decode())
        modified_message = message.decode().upper()
        server_socket.sendto(modified_message.encode(), client_address)
    ```

For the function `socket`, the first parameter indicates the address family;
in particular, `AF_INET` indicates that the underlying network is using IPv4.
The second parameter indicates that the socket is of type `SOCK_DGRAM`, which
means it is a UDP socket. 

For a TCP/IP/UDP socket connection, the send and receive buffer sizes define 
the receive window. The receive window specifies the amount of data that 
can be sent and not received before the send is interrupted. If too much 
data is sent, it overruns the buffer and interrupts the transfer. The 
mechanism that controls data transfer interruptions is referred to as 
flow control. If the receive window size for TCP/IP buffers is too small, 
the receive window buffer is frequently overrun, and the flow control 
mechanism stops the data transfer until the receive buffer is empty.

This buffer size is controlled by `recvfrom(2048)`. 

## Socket programming with TCP

Unlike UDP, TCP is a connection-oriented protocol. This means that before the
client and server can start to send data to each other, they first need to
handshake and establish a TCP connection. 

![tcp-socket](./images/tcp-two-sockets.png)

=== "tcp_client.py"
    ```py
    from socket import *


    server_name = '47.108.238.80'
    server_port = 12000  # make sure you opened this port

    client_socket = socket(AF_INET, SOCK_STREAM)

    try:
        # create the connect 
        client_socket.connect((server_name, server_port))
    except:
        print("connection failed")

    message = input("What's you message: \n")

    client_socket.send(message.encode())

    reply_message = client_socket.recv(1024)

    print('From Server: ', reply_message.decode())

    client_socket.close()
    ```
=== "tcp_server.py"
    ```py
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
    ```

## Opening a port on Linux

Typically, ports identify a specific network service assigned to them. This 
can be changed by manually configuring the service to use a different port, 
but in general, the defaults can be used.

The first 1024 ports (Ports 0-1023) are referred to as well-known port 
numbers and are reserved for the most commonly used services include SSH (port 22), 
HTTP and HTTPS (port 80 and 443), etc. Port numbers above 1024 are referred to as ephemeral ports.

Among ephemeral ports, Port numbers 1024-49151 are called the Registered/User 
Ports. The rest of the ports, 49152-65535 are called as Dynamic/Private Ports.

we could use `ss` command to list listening sockets with an open port.

```bash
ss -lntu
```

It gives the following results.

```bash
Netid        State         Recv-Q        Send-Q                    Local Address:Port                Peer Address:Port       Process        
udp          UNCONN        0             0                    172.25.161.18%eth0:68                       0.0.0.0:*                         
udp          UNCONN        0             0                             127.0.0.1:323                      0.0.0.0:*                         
udp          UNCONN        0             0                         127.0.0.53%lo:53                       0.0.0.0:*                         
udp          UNCONN        0             0                                 [::1]:323                         [::]:*                         
udp          UNCONN        0             0                                     *:55896                          *:*                         
udp          UNCONN        0             0                                     *:19286                          *:*                         
udp          UNCONN        0             0                                     *:22313                          *:*                         
tcp          LISTEN        0             4096                          127.0.0.1:9090                     0.0.0.0:*                         
tcp          LISTEN        0             511                           127.0.0.1:9091                     0.0.0.0:*                         
tcp          LISTEN        0             4096                          127.0.0.1:9092                     0.0.0.0:*                         
tcp          LISTEN        0             4096                      127.0.0.53%lo:53                       0.0.0.0:*                         
tcp          LISTEN        0             128                             0.0.0.0:22                       0.0.0.0:*                         
tcp          LISTEN        0             511                                   *:23426                          *:*                         
tcp          LISTEN        0             4096                                  *:22313                          *:*                         
tcp          LISTEN        0             4096                                  *:19286                          *:*                         
tcp          LISTEN        0             4096                                  *:55896                          *:* 
```

We can verify whether a port is being used or not.

```bash
ss -na | grep :4000
```

If it returns _nothing_, then it means the port is not being used. Otherwise,
it should return the status of the port.

```bash
ss -na | grep :55896

udp      UNCONN                 0                   0               *:55896                *:*                                 
tcp      LISTEN                 0                   4096            *:55896                *:* 
```

Ubuntu has a firewall called `ufw`, which takes care of these rules for ports 
and connections, instead of the old `iptables` firewall. If you are a Ubuntu user, 
you can directly open the port using `ufw`

```sh
sudo ufw allow 4000
```

After opening a port, you could test it with `telnet`.

```bash
telnet 47.108.238.80 12000

Trying 47.108.238.80...
Connected to 47.108.238.80.
Escape character is '^]'.
```

This will open the port `4000`. Now, you could test your `tcp_client.py` and
`tcp_server.py` from your computer and your server. 

## Creating a Web server 

In this section, we will create a web server. Before starting it, please be
aware that __Web is not the Internet__ and please look at the following figure
again. 

![thin-waist](./images/thin-waist.png)

A web server will:

1. create a connection socket when contacted by a client(browser)
2. receive the HTTP request from connection
3. parse the request to determine the specific file 
4. get the requested file from the server's file system 
5. create an HTTP response message 
6. send the response over the TCP connection 

In my server (or host), I put `HelloWorld.html` and `web_server.py` into the
same directory.

```bash
.
├── HelloWorld.html
└── web_server.py
```

`HelloWrold.html` is a very simple HTML file:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>You see me</title>
</head>
<body>
    Hello, World !
</body>
</html>
```

Here is the code of the `web_server.py`:

```python
#import socket module
from socket import *
import sys # In order to terminate the program


#Prepare a sever socket

server_host = ''  # receive all interface
# use localhost for test: server_host = 'localhost'
server_port = 12000  # TCP connection multiplexing 

with socket(AF_INET, SOCK_STREAM) as socket_server:
    # override the port
    socket_server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)  
    socket_server.bind((server_host, server_port))  # bind ip + port
    socket_server.listen(1)
    print("The server is ready to receive")
    while True:
        # create the connection
        connection_socket, addr = socket_server.accept()  
        print(f"Connected by {addr}")
        # connect and send the data 
        try:
            message = connection_socket.recv(1024).decode()
            print(message)
            # the path is the second part of the header
            # GET /HelloWorld.html HTTP/1.1
            file_name = message.split()[1]  # /HelloWorld.html
            file = open(file_name[1:])  # open HelloWorld.html
            html_text = file.read()
            #Send one HTTP header line into socket
            connection_socket.send("HTTP/1.1 200 OK\r\n\r\n".encode())
            # send the content to the client 
            for i in range(0, len(html_text)):
                connection_socket.send(html_text[i].encode())
            connection_socket.send("\r\n".encode())
            connection_socket.close()
        except IOError:
            print("IOError")
            #Send response message for file not found
            connection_socket.send(
                "HTTP/1.1 404 Not Found\r\n\r\n".encode()
                )
            connection_socket.close()
            
sys.exit()  # terminate the program when it stops serving 
```

Now, I will visit the website from the browser. 

![client](./images/browser-visit.png)

On my server, after _listening_ the request, I got the following requests:

```bash
root@iZ2vc0ibe0pndu24p4g2j7Z:~/cs144/website# python3 web_server.py
The server is ready to receive
Connected by ('8*.1*9.*6.*8', 50722)  # my address and port 

# request by the client's browser 
GET /HelloWorld.html HTTP/1.1
Host: 47.108.238.80:12000
Connection: keep-alive
Cache-Control: max-age=0
Upgrade-Insecure-Requests: 1
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.124 Safari/537.36 Edg/102.0.1245.41
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
Accept-Encoding: gzip, deflate
Accept-Language: en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7
```

On my browser, I got the file I requested.

![hello-world](./images/browser-response.png)


## UDP Pinger


=== "tcp_ping_client.py"
    ```py
    """
    This ping program send 10 ping messages to the target server over UDP
        - print the Round Trip Time when the corresponding pong 
        message is returned
        - timeout = 1 second when no data is received 
    """

    import time
    from socket import *

    server_ip = '47.108.238.80'
    server_port = 12000

    with socket(AF_INET, SOCK_DGRAM) as client_socket:
        client_socket.settimeout(1)
        # ping ten times
        for i in range(10):
            t0 = time.time()
            message = 'PING ' + str(i + 1) + " " + str(time.strftime("%H:%M:%S"))
            client_socket.sendto(message.encode(), (server_ip, server_port))
            
            try:
                data, server = client_socket.recvfrom(1024)
                data = data.decode()
                t1 = time.time()
                rtt = t1 - t0  # round trip time 
                print(f"Data received - {data}, Round Trip Time: {rtt}")
            except timeout:
                print("REQUEST Time Out")
    ```

=== "udp_ping_server.py"
    ```py
    import random
    from socket import *


    # Create a UDP socket
    serverSocket = socket(AF_INET, SOCK_DGRAM)
    # Assign IP address and port number to socket
    serverSocket.bind(('', 12000))

    while True:
        # Generate random number in the range of 0 to 10
        rand = random.randint(0, 10)
        # Receive the client packet along with the address it is coming from
        message, address = serverSocket.recvfrom(1024)
        # Capitalize the message from the client
        message = message.upper()
        # If rand is less is than 4, we consider the packet lost and do not respond
        if rand < 4:
            continue
        # Otherwise, the server responds
        serverSocket.sendto(message, address)
    ```


## Mail Client


## Web Proxy