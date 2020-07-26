from multiprocessing import Process, Queue
import numpy as np

import OpenGL.GL as gl
import pangolin

"""
import sys
sys.path.append('./pangolin.cpython-36m-x86_64-linux-gnu.so')
"""

class Point3d(object):
    # It represents a 3d point obtain through the triangulation procedure
    def __init__(self, mapp, loc):
        self.location = loc
        self.frames = []
        self.idxs = []
        self.id = len(mapp.points)
        mapp.points.append(self)

    def add_frame_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)

class Map(object):
    def __init__(self):
        self.frames = []       
        self.points = []
        self.q = Queue()

        self.currState = None

        process = Process(target=self.update_map_thread, args=(self.q,))
        process.daemon = True
        process.start()

    def init_thread(self):
        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)
        
         # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin.ModelViewLookAt(0, -10, -8, 0, 0, 0, 0, -1, 0))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(self.handler)

    def update_map_thread(self, q):
        self.init_thread()
        while 1:
                self.render_map(q)


    def render_map(self, q):
        if (self.currState is None) or (not q.empty()):
            self.currState = self.q.get()
            
            # Extract points and poses
            pts = np.array(self.currState[1])
            
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            self.dcam.Activate(self.scam)

            #gl.glPointSize(10)
            gl.glColor3f(1.0, 0.0, 0.0)
            pangolin.DrawCameras(self.currState[0])

            gl.glPointSize(2)
            gl.glColor3f(0.0, 1.0, 0.0)
            pangolin.DrawPoints(pts)

            pangolin.FinishFrame()


    def display_map(self):
        poses, points = [], []
        
        for f in self.frames:
            poses.append(f.pose)
        
        for p in self.points:
            points.append(p.location)

        self.q.put((poses, points))
