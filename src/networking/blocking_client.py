import socket
import sys
import time


def main() -> None:
    host = socket.gethostname()
    port = 12000

    # create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        while True:
            sock.connect((host, port))
            while True:
                data = str.encode(sys.argv[1])
                sock.send(data)
                time.sleep(0.5)
                

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please provide message"
    main()