#!./usr/bin/env python3

from display import displayVideo


if __name__ == "__main__":
    path = './files/drivingCar.mp4'

    dp = displayVideo()
    dp.display(path)    
    