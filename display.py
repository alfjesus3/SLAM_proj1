import time
import cv2
import sdl2
import sdl2.ext


class displayVideo():
    def __init__(self):
        self.W = 1920 // 2
        self.H = 1080 // 2

        self.window = sdl2.ext.Window("SLAM", size=(self.W,self.H))
        self.window.show()

    def process_frame(self, img):
        img = cv2.resize(img, (self.W,self.H))

        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)

        surf = sdl2.ext.pixels3d(self.window.get_surface())
        surf[:, :, 0:3] = img.swapaxes(0,1)

        self.window.refresh()
        
        """
        print(img.shape)
        print(img)
        time.sleep(1)
        """