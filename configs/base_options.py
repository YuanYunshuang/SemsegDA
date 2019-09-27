import os


class BaseOptions():
    def __init__(self):
        self.visual = VisualOptions()
        self.log = LogOptions()


class VisualOptions():
    def __init__(self):
        self.path = None
        self.display_freq = 1    # frequency of showing training results on screen
        self.display_ncols=4       #if positive, display all images in a single visdom web panel with certain number of images per row.'
        self.display_id = 1        #'window id of the web display')
        self.display_server = "http://localhost"
        self.display_env = 'main'
        self.display_port = 8023
        self.update_html_freq = 1000
        self.display_winsize = 512
        self.web_dir = '/home/robotics/SemsegDA/callbacks/web'
        self.image_dir = '/home/robotics/SemsegDA/callbacks/images'
        if not os.path.exists(self.web_dir):
            os.mkdir(self.web_dir)
        if not os.path.exists(self.image_dir):
            os.mkdir(self.image_dir)

class LogOptions():
    def __init__(self):
        self.path = '/home/robotics/SemsegDA/callbacks/log.txt'
