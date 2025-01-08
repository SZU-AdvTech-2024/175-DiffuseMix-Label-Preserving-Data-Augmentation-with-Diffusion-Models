import torch
import numpy
from ..utils import *
from ..gradient.mifgsm import MIFGSM


class SPLINE(MIFGSM):
    """
    DeCowA(Warping Attack)
    'Boosting Adversarial Transferability across Model Genus by Deformation-Constrained Warping (AAAI 2024)'(https://arxiv.org/abs/2402.03951)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        mesh_width: the number of control points in width.
        mesh_height: the number of control points in height.
        noise_scale: random noise strength.
        num_warping: the number of warping transformation samples.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data.
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1., mesh_width=3, mesh_height=3,
                 rho=0.01,
                 num_warping=20, noise_scale=2, targeted=False, random_start=False, norm='linfty',
                 loss='crossentropy', device=None, attack='SPLINE', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_warping = num_warping
        self.noise_scale = noise_scale
        self.mesh_width = mesh_width
        self.mesh_height = mesh_height
        self.epsilon = epsilon
        self.rho = rho

    def vwt(self, x, noise_map):  # 扭曲变换函数，使用双三次插值替换 TPS
        # 获取输入图像的维度：批量大小 n、通道数 c、高度 h、宽度 w
        n, c, h, w = x.size()
        device = x.device

        # 1. 生成控制点网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, self.mesh_height, device=device),
            torch.linspace(-1, 1, self.mesh_width, device=device),
            indexing='ij'  # 确保维度顺序正确
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [mesh_height, mesh_width, 2]

        # 2. 添加噪声得到扭曲后的控制点
        # noise_map 的形状应为 [mesh_height - 2, mesh_width - 2, 2]
        # 创建一个与 grid 形状相同的扰动张量
        perturbation = torch.zeros_like(grid)
        perturbation[1:-1, 1:-1, :] = noise_map  # 仅对非边界点添加噪声

        # 计算扭曲后的控制点
        warped_grid = grid + perturbation  # [mesh_height, mesh_width, 2]

        # 3. 使用双三次插值计算密集网格
        # 需要将控制点和对应的位移展平成网格格式

        # 生成用于插值的密集坐标网格
        dense_grid_y, dense_grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        dense_grid = torch.stack([dense_grid_x, dense_grid_y], dim=-1)  # [h, w, 2]

        # 计算位移场
        # 首先，计算控制点的位移
        displacement = warped_grid - grid  # [mesh_height, mesh_width, 2]

        # 使用 `grid_sample` 进行插值，需要将位移场转换为 [1, 2, mesh_height, mesh_width]
        displacement = displacement.permute(2, 0, 1).unsqueeze(0)  # [1, 2, mesh_height, mesh_width]

        # 将密集网格坐标标准化到 [0, 1]，以匹配 `interpolate` 的要求
        grid_normalized = (dense_grid + 1) / 2  # 将 [-1, 1] 映射到 [0, 1]

        # 使用 `interpolate` 对位移场进行双三次插值，得到密集位移场
        displacement_dense = torch.nn.functional.interpolate(
            displacement, size=(h, w), mode='bicubic', align_corners=True
        )  # [1, 2, h, w]

        # 将位移场转换回 [h, w, 2]
        displacement_dense = displacement_dense.squeeze(0).permute(1, 2, 0)  # [h, w, 2]

        # 4. 应用位移场
        warped_dense_grid = dense_grid + displacement_dense  # [h, w, 2]

        # 将网格从 [h, w, 2] 转换为 [n, h, w, 2]
        warped_dense_grid = warped_dense_grid.unsqueeze(0).expand(n, -1, -1, -1)

        # 使用 `grid_sample` 应用扭曲变换
        warped_x = torch.nn.functional.grid_sample(
            x,
            warped_dense_grid,
            mode='bilinear',
            padding_mode='reflection',
            align_corners=True
        )

        return warped_x

    def vwt(self, x, noise_map):  # 扭曲变换函数，使用双三次插值替换 TPS
        n, c, h, w = x.size()
        device = x.device

        # 1. 生成控制点网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, self.mesh_height, device=device),
            torch.linspace(-1, 1, self.mesh_width, device=device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [mesh_height, mesh_width, 2]

        # 2. 添加噪声得到扭曲后的控制点
        perturbation = torch.zeros_like(grid)
        perturbation[1:-1, 1:-1, :] = noise_map  # 仅对非边界点添加噪声

        # 计算扭曲后的控制点
        warped_grid = grid + perturbation  # [mesh_height, mesh_width, 2]

        # 3. 使用双三次插值计算密集位移场
        # 控制点位移
        displacement = warped_grid - grid  # [mesh_height, mesh_width, 2]
        displacement = displacement.permute(2, 0, 1).unsqueeze(0)  # [1, 2, mesh_height, mesh_width]

        # 使用 interpolate 进行插值
        displacement_dense = torch.nn.functional.interpolate(
            displacement, size=(h, w), mode='bicubic', align_corners=True
        )  # [1, 2, h, w]

        displacement_dense = displacement_dense.squeeze(0).permute(1, 2, 0)  # [h, w, 2]

        # 生成密集坐标网格
        dense_grid_y, dense_grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        dense_grid = torch.stack([dense_grid_x, dense_grid_y], dim=-1)  # [h, w, 2]

        # 4. 应用位移场
        warped_dense_grid = dense_grid + displacement_dense  # [h, w, 2]
        warped_dense_grid = warped_dense_grid.unsqueeze(0).expand(n, -1, -1, -1)

        # 使用 grid_sample 应用扭曲变换
        warped_x = torch.nn.functional.grid_sample(
            x,
            warped_dense_grid,
            mode='bilinear',
            padding_mode='reflection',
            align_corners=True
        )

        return warped_x

    def update_noise_map(self, x, label):
        x.requires_grad = False
        noise_map = (torch.rand([self.mesh_height - 2, self.mesh_width - 2, 2], device=self.device) - 0.5) * self.noise_scale
        for _ in range(1):
            noise_map.requires_grad = True
            vwt_x = self.vwt(x, noise_map)
            logits = self.get_logits(vwt_x)
            loss = self.get_loss(logits, label)
            grad = torch.autograd.grad(loss, noise_map)[0]
            noise_map = noise_map.detach() - self.rho * grad
        return noise_map.detach()

    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            grads = 0
            for _ in range(self.num_warping):
                # Obtain the data after warping
                adv = (data + delta).clone().detach()
                noise_map_hat = self.update_noise_map(adv, label)
                vwt_x = self.vwt(data + delta, noise_map_hat)

                # Obtain the output
                logits = self.get_logits(vwt_x)

                # Calculate the loss
                loss = self.get_loss(logits, label)

                # Calculate the gradients on delta
                grad = self.get_grad(loss, delta)
                grads += grad

            grads /= self.num_warping

            # Calculate the momentum
            momentum = self.get_momentum(grads, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
