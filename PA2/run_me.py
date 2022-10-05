from threading import Thread

from agent import main
from server import execute

if __name__ == "__main__":
    th1 = Thread(target=main)
    th1.start()

    th2 = Thread(target=execute)
    th2.start()
