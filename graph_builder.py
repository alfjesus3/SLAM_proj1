from multiprocessing import Process, Queue
import numpy as np

import OpenGL.GL as gl
from build.pangolin import pangolin

class Map(object):
    def __init__(self):
        self.frames = []       
        self.points = []
        self.q = Queue()

        self.currState = None

        process = Process(target=self.init_map_thread, args=(self.q,))
        process.daemon = True
        process.start()


    def init_map_thread(self, q):
        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)
        
         # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(self.handler)
        
        while 1:
                self.render_map(q)


    def render_map(self, q):
        if (self.currState is not None) or (not self.q.empty()):
            self.currState = self.q.get()
            
            # Extract points and poses
            poses = np.array([po[:3, 3] for po in self.currState[0]])
            pts = np.array(self.currState[1])
            
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            self.dcam.Activate(self.scam)

            gl.glPointSize(10)
            gl.glColor3f(1.0, 0.0, 0.0)
            pangolin.DrawPoints(poses)

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
