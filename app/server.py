import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1234))
s.listen(5)

while True:
    client_socket, address = s.accept()
    print(f"connect feom {address} has been established")
    client_socket.send(bytes("welcome to the server", "utf-8"))
    client_socket.close()