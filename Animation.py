import matplotlib.animation as manimation


class Animation:
    def __init__(self, fig, file_name):
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        self.writer = FFMpegWriter(fps=1, metadata=metadata)
        self.writer.setup(fig, file_name+".mp4", 100)

    def add_frame(self):
        self.writer.grab_frame()
