
from utils.batch_statistics import StatsRecorder


class ActivationStatistics:
    """
    Calculate mean and std for each channel in each layer of activations.
    """
    def __init__(self, cfg, model, dataset):
        self.cfg = cfg
        self.layers = model
        self.dataset = dataset
        self.stats = None

    def cal_mean_std(self):
        for images, labels in self.dataset:
            self.model.set_input(images, labels)
            self.model.forward()
            if self.stats == None:
                self.stats = [StatsRecorder() for i in range(len(self.model.hooks))]
            for i, hook in enumerate(self.model.hooks):
                activation = hook.output
                b, c, h, w = activation.shape
                activation = activation.transpose(0,1).reshape(c, -1).transpose(0,1)
                self.stats[i].update(activation)

    def get_mean_std(self, layer, channel):
        mean = self.stats.get_mean(layer, channel)
        std = self.stats.get_std(layer, channel)

        return mean, std


class StatsRecoderStack:
    def __init__(self):
        self.layerSRs = []

    def add_layerSR(self, channels):
        self.layerSRs.append([StatsRecorder() for i in range(channels)])

    def update_layerSR(self, layer_index, activation):
        c, l = activation.shape
        if len(self.layerSRs) < layer_index + 1:
            self.layerSRs.append([StatsRecorder() for i in range(c)])

        for i in range(c):
            self.layerSRs[layer_index][i].update(activation[i,:])

    def get_mean(self, layer, channel):
        return self.layerSRs[layer][channel].mean

    def et_std(self, layer, channel):
        return self.layerSRs[layer][channel].std
