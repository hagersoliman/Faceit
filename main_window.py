"""
In this example, we demonstrate how to create simple camera viewer using Opencv3 and PyQt5

Author: Berrouba.A
Last edited: 21 Feb 2018
"""

# import system module
import sys

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer

# import Opencv module
import cv2

from ui_main_window import *

from face2face import *
import zmq

class MainWindow(QWidget):
    # class constructor
    def __init__(self,socket,generator,kp_detector):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.logo_setup()

        self.socket=socket
        self.generator=generator
        self.kp_detector=kp_detector

        self.first_time_to_send=1
        self.first_time_to_recieve=1

        self.init_source=None
        self.init_kp=None
        
        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.ui.control_bt.clicked.connect(self.controlTimer)
        

    def logo_setup(self):
        logo=cv2.imread("faceit.png")
        logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
            
        # get image infos
        height, width, channel = logo.shape
        step = channel * width
        # create QImage from image
        
        qImg = QImage(logo.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.image_label3.setPixmap(QPixmap.fromImage(qImg.scaled(100, 100, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)))

    # view camera
    def viewCam(self):

        ###################sending######################################
	
        img,frame = self.vid.read()

        if frame is not None:

            ###########################ui##########################################
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # get image infos
            height, width, channel = frame.shape
            step = channel * width
            # create QImage from image
            
            qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            # show image in img_label
            self.ui.image_label2.setPixmap(QPixmap.fromImage(qImg.scaled(300, 300, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)))
            

            #######################################################################
            if self.first_time_to_send==1:
                
                dic={"image":frame}
            
                self.socket.send_pyobj(dic) 
                
                self.first_time_to_send=0
            
            else:
                
                resized_frame = torch.tensor( resize(frame, (256, 256))[..., :3][np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                resized_frame = resized_frame.cuda()
                key_points= self.kp_detector(resized_frame)
               
                self.socket.send_pyobj(key_points) 


        ###################recieving#####################################
        try:
            
            if self.first_time_to_recieve==1:
                
                dic=self.socket.recv_pyobj() 
               
                source_image=dic["image"]
                source_image = resize(source_image, (256, 256))[..., :3]
                source_frame = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                source_frame = source_frame.cuda()
                #sourceee
                self.init_source=source_frame

                
                kp_source = self.kp_detector(source_frame)

                
                #source
                self.init_kp=kp_source

                self.first_time_to_recieve=0

             
            else:
              
                kp_driving=self.socket.recv_pyobj()
                
                kp_norm = get_new_kp(kp_source=self.init_kp, kp_driving=kp_driving,
                                    kp_driving_initial=self.init_kp, use_relative_movement=True,
                                    use_relative_jacobian=True, adapt_movement_scale=True)
                out = self.generator(self.init_source, kp_source=self.init_kp, kp_driving=kp_norm)
                generated_frame = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0].copy()

                
                ################################ui##############################################
                generated_frame=generated_frame*255
                generated_frame=generated_frame.astype(np.uint8)
                
                # convert image to RGB format
                #n = cv2.cvtColor(generated_frame, cv2.COLOR_BGR2RGB)
            

                height, width, channel = generated_frame.shape

                step = channel * width
                # create QImage from image
              
                qImg = QImage(generated_frame.data, width, height, step, QImage.Format_RGB888)
                # show image in img_label
               
                self.ui.image_label.setPixmap(QPixmap.fromImage(qImg.scaled(300, 300, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)))
 
                ############################################################################

        except :
            print("nothin to receive now or ther is some error in the try block")
	

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.vid = cv2.VideoCapture("crop.mp4")
            # start timer
            self.timer.start(0)
            # update control_bt text
            self.ui.control_bt.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.vid.release()
            # update control_bt text
            
            self.ui.control_bt.setText("Start")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    who=input("are you server or client ")
    if who=="s":
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        port="5555"
        socket.bind("tcp://*:"+port)
        socket.RCVTIMEO = 1000 
    elif who=="c":
        port = "5555"
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        socket.connect("tcp://localhost:"+port)
        socket.RCVTIMEO = 1000 
        

    generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',checkpoint_path='vox-cpk.pth.tar')

    # create and show mainWindow
    mainWindow = MainWindow(socket,generator,kp_detector)
    mainWindow.show()

    sys.exit(app.exec_())