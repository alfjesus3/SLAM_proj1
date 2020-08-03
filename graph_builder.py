from multiprocessing import Process, Queue
import numpy as np
import g2o
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
        frame.pts[idx] = self
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

    ## Optimizer ##
    def optimize(self, max_iters):
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(solver)
        
        robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))

        # adding frames to graph
        for f in self.frames:
            sbacam = g2o.SBACam(g2o.SE3Quat(f.pose[0:3, 0:3], f.pose[0:3, 3]))
            sbacam.set_cam(f.T[0][0], f.T[1][1], f.T[2][0], f.T[2][1], 1.0)

            v_se3 = g2o.VertexCam()
            v_se3.set_id(f.id)
            v_se3.set_estimate(sbacam)
            v_se3.set_fixed(f.id == 0)
            opt.add_vertex(v_se3)
        
        # add points to graph
        for p in self.points:
            pt = g2o.VertexSBAPointXYZ()
            pt.set_id(p.id + 0x10000)
            pt.set_estimate(p.location[0:3])
            pt.set_marginalized(True)
            pt.set_fixed(False)
            opt.add_vertex(pt)

            # add connections between frames that contain a point in the graph
            for f in p.frames:
                edge = g2o.EdgeProjectP2MC()
                edge.set_vertex(0, pt)
                edge.set_vertex(1, opt.vertex(f.id))
                uv = f.kpts[f.pts.index(p)]
                edge.set_measurement(uv)
                edge.set_information(np.eye(2))
                edge.set_robust_kernel(robust_kernel)
                opt.add_edge(edge)

        opt.set_verbose(True)
        opt.initialize_optimization()
        opt.optimize(max_iters)

