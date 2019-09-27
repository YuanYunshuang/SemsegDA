import torch
from torch import nn
import functools
from loss.cross_entropy import cross_entropy2d
from optimizers import get_optimizer
from schedulers import get_scheduler



class DRCN(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, cfg):
        """Construct a Unet generator
        Parameters in cfg:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(DRCN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_nc = cfg['model']['input_nc']
        output_nc = (cfg['model']['output_nc1'], cfg['model']['output_nc2'])
        num_downs = cfg['model']['num_downs']
        ngf = cfg['model']['ngf']
        norm_layer = nn.BatchNorm2d if cfg['model']['norm_layer']=='batch' else nn.InstanceNorm2d
        use_dropout = cfg['model']['use_dropout']
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

        self.out1 = None
        self.out2 = None
        self.criterionPth1 = cross_entropy2d
        self.criterionPth2 = torch.nn.L1Loss()
        self.loss1 = None
        self.loss2 = torch.Tensor([0.])
        self.optimizer = get_optimizer(self.parameters(), cfg)
        if cfg["training"]["lr_schedule"] is not None:
            self.scheduler = get_scheduler(self.optimizer, cfg["training"]["lr_schedule"])
        self.lam = cfg["training"]["loss"]["lambda"]

    def set_input(self, image, label):
        self.A = image
        self.B = label

    def forward(self):
        """Standard forward"""
        out = self.model(self.A.to(self.device))
        self.out1 = out[0]
        self.out2 = out[1]

    def backward(self):
        self.loss1 = self.lam * self.criterionPth1(self.out1, self.B.to(self.device))
        #self.loss2 = (1 - self.lam) * self.criterionPth2(self.out2, self.A.to(self.device))
        loss = self.loss1 #+ self.loss2
        loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def inference(self):
        self.forward()




class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)
        if outermost:
            assert len(outer_nc) == 2
            upconv1 = nn.ConvTranspose2d(inner_nc * 2, outer_nc[0],
                                        kernel_size=4, stride=2,
                                        padding=1)
            upconv2 = nn.ConvTranspose2d(inner_nc * 2, outer_nc[1],
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv, submodule]
            up1 = [uprelu, upconv1, nn.Softmax2d()]
            up2 = [uprelu, upconv2, nn.Tanh()]

        elif innermost:
            assert isinstance(outer_nc, int)
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up1 = [uprelu, upconv, upnorm]
            up2 = up1

        else:
            assert isinstance(outer_nc, int)
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm, submodule]
            up1 = [uprelu, upconv, upnorm]

            if use_dropout:
                up1 = up1 + [nn.Dropout(0.5)]
            up2 = up1

        self.down = nn.Sequential(*down)
        self.up1 = nn.Sequential(*up1)
        self.up2 = nn.Sequential(*up2)

    def forward(self, x):
        if self.outermost:
            out = self.down(x)
            out1 = self.up1(out[0])
            out2 = self.up2(out[1])
        elif self.innermost:
            out = self.down(x)
            out1 = torch.cat([x, self.up1(out)], 1)
            out2 = torch.cat([x, self.up2(out)], 1)
        else:   # add skip connections
            out = self.down(x)
            out1 = torch.cat([x, self.up1(out[0])], 1)
            out2 = torch.cat([x, self.up2(out[1])], 1)
        return out1, out2