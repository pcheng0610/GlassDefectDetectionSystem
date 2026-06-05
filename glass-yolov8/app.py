from pythonWeb import create_app
import multiprocessing
import os
import socket
import sys

os.environ['PYTHONIOENCODING'] = 'utf-8'

def main():
    app = create_app()
    socket.getfqdn = lambda x: 'localhost'
    app.run(use_reloader=False, debug=True, port=5002)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
