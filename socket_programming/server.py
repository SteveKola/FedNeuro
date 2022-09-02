import socket


def server_program():
    host = socket.gethostname()
    port = 5000  # initiate port no above 1024

    server_socket = socket.socket()  # get instance
    server_socket.bind((host, port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(3)
    conn, address = server_socket.accept()
    print("Connection from: " + str(address))
    while True:
        data = conn.recv(
            1024
        ).decode()  # it won't accept data packet greater than 1024 bytes
        if not data:
            break
        print("from connected user: " + str(data))
        data = input(" -> ")
        conn.send(data.encode())  # send data to the client

    conn.close()


if __name__ == "__main__":
    server_program()
