from multiprocessing import Process, Queue
import numpy as np

import OpenGL.GL as gl
import pangolin


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
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 1000),
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
            self.currState = q.get()
            
        # Extract points and poses
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        # Draw poses
        #gl.glPointSize(10)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawCameras(self.currState[0])
        
        # Draw points
        gl.glPointSize(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(self.currState[1])

        pangolin.FinishFrame()


    def display(self):
        poses, points = [], []
        
        for f in self.frames:
            poses.append(f.pose)
        
        for p in self.points:
            points.append(p.location)

        self.q.put((np.array(poses), np.array(points)))
