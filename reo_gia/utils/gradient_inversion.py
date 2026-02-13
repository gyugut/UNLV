"""
gradient_inversion.py
=====================
inversefed 라이브러리의 핵심 공격 로직을 분석하여 재구현한 모듈.
외부 의존성 없이 PyTorch만으로 동작하며, 어떤 모델이든 (CNN, Transformer) 사용 가능.

원본: https://github.com/JonasGeiping/invertinggradients
논문: Geiping et al., "Inverting Gradients", NeurIPS 2020

재현 대상 논문 설정:
  Zhang et al., "How Does a Deep Learning Model Architecture Impact Its Privacy?"
  - cost_fn: cosine similarity
  - optimizer: Adam
  - lr: 0.1, lr_decay: True
  - max_iterations: 3000
  - total_variation: 1e-4
  - signed gradients: True
  - boxed constraint: True
"""

import torch
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Optional


# =============================================================================
# 1. 데이터셋 설정 (Dataset Configs)
# =============================================================================
# inversefed.consts에 해당하는 부분

DATASET_CONFIGS = {
    'cifar10': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std':  (0.2023, 0.1994, 0.2010),
        'img_shape': (3, 32, 32),
        'num_classes': 10,
    },
    'imagenet': {
        'mean': (0.485, 0.456, 0.406),
        'std':  (0.229, 0.224, 0.225),
        'img_shape': (3, 224, 224),
        'num_classes': 1000,
    },
}

# 1번 논문 Table 1의 공격 하이퍼파라미터
DEFAULT_ATTACK_CONFIG = {
    'cost_fn':         'sim',     # cosine similarity (inversefed의 'sim')
    'optimizer':       'adam',    # Adam optimizer
    'lr':              0.1,       # 학습률
    'lr_decay':        True,      # 학습률 감쇠 여부
    'max_iterations':  3000,      # 최대 반복 횟수
    'total_variation': 1e-4,      # TV 정규화 가중치
    'signed':          False,      # signed gradient 사용 (adversarial attack 기법)
    'boxed':           True,      # 유효 범위 내로 클램핑
    'restarts':        1,         # 재시작 횟수
    'init':            'randn',   # 초기화 방법: 'randn' 또는 'rand'
}


# =============================================================================
# 2. 비용 함수 (Cost Functions)
# =============================================================================
# inversefed/reconstruction_algorithms.py의 reconstruction_costs() 에 해당

def cost_fn_cosine_sim(trial_gradients: List[torch.Tensor],
                       target_gradients: List[torch.Tensor]) -> torch.Tensor:
    """
    Cosine similarity 기반 비용 함수.

    inversefed 원본 구현 분석:
    ─────────────────────────
    원본은 per-layer가 아닌 **전체 gradient를 하나의 벡터로 보고** cosine similarity를 계산.
    구체적으로:
        dot_product = sum(trial[i] * target[i] for all layers)
        norm_trial  = sqrt(sum(trial[i]^2 for all layers))
        norm_target = sqrt(sum(target[i]^2 for all layers))
        cost = 1 - dot_product / (norm_trial * norm_target)

    이는 모든 gradient를 flatten → concat → cosine_similarity 하는 것과 동일.

    Returns:
        cost: 0이면 완벽히 일치, 2이면 정반대 방향
    """
    dot_product = torch.tensor(0.0, device=trial_gradients[0].device)
    norm_trial_sq = torch.tensor(0.0, device=trial_gradients[0].device)
    norm_target_sq = torch.tensor(0.0, device=trial_gradients[0].device)

    for tg, ig in zip(trial_gradients, target_gradients):
        dot_product += (tg * ig).sum()
        norm_trial_sq += tg.pow(2).sum()
        norm_target_sq += ig.pow(2).sum()

    cost = 1.0 - dot_product / (norm_trial_sq.sqrt() * norm_target_sq.sqrt() + 1e-12)
    return cost


def cost_fn_l2(trial_gradients: List[torch.Tensor],
               target_gradients: List[torch.Tensor]) -> torch.Tensor:
    """
    L2 (Euclidean) 비용 함수. DLG/iDLG 논문에서 사용한 원본 방식.
    비교 실험용으로 포함.
    """
    cost = torch.tensor(0.0, device=trial_gradients[0].device)
    for tg, ig in zip(trial_gradients, target_gradients):
        cost += (tg - ig).pow(2).sum()
    return cost


COST_FUNCTIONS = {
    'sim': cost_fn_cosine_sim,
    'l2':  cost_fn_l2,
}


# =============================================================================
# 3. 정규화 (Regularization)
# =============================================================================
# inversefed에서 total_variation으로 사용하는 부분

def total_variation1(x: torch.Tensor) -> torch.Tensor:
    """
    Total Variation 정규화.
    이미지의 인접 픽셀 간 차이를 최소화하여 노이즈를 억제.

    inversefed 원본:
        TV(x) = sum(|x[..., i+1, j] - x[..., i, j]|^2)
              + sum(|x[..., i, j+1] - x[..., i, j]|^2)

    Args:
        x: (B, C, H, W) 텐서
    Returns:
        스칼라 TV 값
    """
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]  # 세로 방향 차이
    dy = x[:, :, :, 1:] - x[:, :, :, :-1]  # 가로 방향 차이
    return dx.pow(2).sum() + dy.pow(2).sum()

def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Original inversefed implementation (L1 + mean)."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


# =============================================================================
# 4. Gradient 추출 (Gradient Extraction)
# =============================================================================

def extract_gradients(model: torch.nn.Module,
                      images: torch.Tensor,
                      labels: torch.Tensor,
                      loss_fn: torch.nn.Module = None) -> List[torch.Tensor]:
    """
    모델에 이미지를 입력하고, loss에 대한 파라미터 gradient를 추출.
    서버가 받는 gradient에 해당.

    Args:
        model:  victim 모델 (eval 모드)
        images: (B, C, H, W) 입력 이미지
        labels: (B,) 정답 라벨
        loss_fn: 손실 함수 (기본: CrossEntropyLoss)
    Returns:
        gradient 리스트 (각 파라미터에 대응)
    """
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()

    model.zero_grad()
    output = model(images)
    loss = loss_fn(output, labels)
    gradients = torch.autograd.grad(loss, model.parameters())
    return [g.detach().clone() for g in gradients]


# =============================================================================
# 5. 핵심 공격 루프 (Gradient Inversion Attack)
# =============================================================================
# inversefed/reconstruction_algorithms.py의 GradientReconstructor 에 해당

class GradientInversionAttack:
    """
    Gradient Inversion Attack 구현체.

    inversefed.GradientReconstructor를 분석하여 재구현.
    어떤 PyTorch 모델이든 (CNN, Transformer 등) 동작.

    사용법:
        attack = GradientInversionAttack(model, dataset='imagenet', config={...})
        reconstructed, stats = attack.reconstruct(target_gradients, labels)
    """

    def __init__(self,
                 model: torch.nn.Module,
                 dataset: str = 'imagenet',
                 config: dict = None,
                 device: torch.device = None):
        """
        Args:
            model:   공격 대상 모델 (eval 모드여야 함)
            dataset: 'cifar10' 또는 'imagenet' (정규화 상수 결정)
            config:  공격 하이퍼파라미터 (None이면 논문 기본값 사용)
            device:  연산 장치
        """
        self.model = model
        self.model.eval()

        # 설정 병합 (사용자 지정 > 기본값)
        self.config = {**DEFAULT_ATTACK_CONFIG}
        if config is not None:
            self.config.update(config)

        # 장치 설정
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device

        # 데이터셋 정규화 상수
        ds_config = DATASET_CONFIGS[dataset]
        self.dm = torch.tensor(ds_config['mean'], device=self.device).view(1, 3, 1, 1)
        self.ds = torch.tensor(ds_config['std'], device=self.device).view(1, 3, 1, 1)
        self.img_shape = ds_config['img_shape']

        # 비용 함수
        self.cost_fn = COST_FUNCTIONS[self.config['cost_fn']]

        # 손실 함수
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _initialize_dummy(self, num_images: int = 1) -> torch.Tensor:
        """
        더미 이미지 초기화.

        inversefed 분석:
        - 'randn': 표준 정규분포 (논문 기본값)
        - 'rand':  [0, 1] 균일분포
        - boxed=True일 때, 나중에 유효 범위로 클램핑
        """
        shape = (num_images, *self.img_shape)

        if self.config['init'] == 'randn':
            dummy = torch.randn(shape, device=self.device, requires_grad=True)
        elif self.config['init'] == 'rand':
            dummy = torch.rand(shape, device=self.device, requires_grad=True)
        else:
            dummy = torch.randn(shape, device=self.device, requires_grad=True)

        return dummy

    def _get_valid_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        boxed constraint를 위한 유효 범위 계산.

        inversefed 분석:
        원본 이미지 범위 [0, 1]을 정규화 공간으로 변환:
          lower = (0 - mean) / std
          upper = (1 - mean) / std
        이 범위 밖의 값은 실제 이미지에서 불가능한 값이므로 클램핑.
        """
        lower = (0.0 - self.dm) / self.ds  # 정규화 공간에서의 최소값
        upper = (1.0 - self.dm) / self.ds  # 정규화 공간에서의 최대값
        return lower, upper

    def reconstruct(self,
                    target_gradients: List[torch.Tensor],
                    labels: torch.Tensor,
                    num_images: int = 1,
                    img_shape: Tuple[int, ...] = None
                    ) -> Tuple[torch.Tensor, Dict]:
        """
        Gradient Inversion 공격 실행.

        inversefed.GradientReconstructor.reconstruct()를 재구현.

        핵심 알고리즘:
        ──────────────
        1. 더미 이미지 x를 랜덤 초기화
        2. 반복:
           a) x를 모델에 통과 → dummy gradient 계산
           b) cost = cosine_similarity_cost(dummy_grad, target_grad)
           c) cost += TV_weight * TV(x)
           d) cost를 x에 대해 미분 (2차 미분 필요)
           e) signed gradient로 x 업데이트 (adversarial attack 기법)
           f) x를 유효 범위로 클램핑 (boxed constraint)
        3. 여러 번 재시작하여 최적 결과 선택

        Args:
            target_gradients: 서버가 수신한 gradient (공격 재료)
            labels: 정답 라벨 (iDLG로 복원 가능하므로 known 가정)
            num_images: 복원할 이미지 수 (기본 1)
            img_shape: 이미지 형태 (None이면 데이터셋 기본값)
        Returns:
            (복원된 이미지, 통계 딕셔너리)
        """
        if img_shape is not None:
            self.img_shape = img_shape

        cfg = self.config
        best_cost = float('inf')
        best_result = None

        for restart_idx in range(cfg['restarts']):
            # ─── 초기화 ───
            x = self._initialize_dummy(num_images)
            # 박스 미리 계산
            lower, upper = self._get_valid_bounds()

            # ─── Optimizer 설정 ───
            # inversefed: Adam with lr, but actual update uses signed gradient
            optimizer = torch.optim.Adam([x], lr=cfg['lr'])

            # ─── LR Scheduler ───
            # inversefed: MultiStepLR, milestones at [max_iter*0.5, max_iter*0.75, ...]
            if cfg['lr_decay']:
                max_iter = cfg['max_iterations']
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[
                        max_iter // 2.667,
                        max_iter // 1.6,
                        max_iter // 1.142,
                    ],
                    gamma=0.1
                )
            else:
                scheduler = None

            # ─── 공격 루프 ───
            history = []
            for iteration in range(cfg['max_iterations']):

                # --- (a) Closure: dummy gradient 계산 + cost 계산 ---
                # inversefed는 closure 패턴을 사용하지만,
                # Adam + signed gradient 조합이라 직접 backward 호출이 더 명확함

                optimizer.zero_grad()
                self.model.zero_grad()

                # 더미 이미지 → 모델 → loss → dummy gradient
                dummy_output = self.model(x)
                dummy_loss = self.loss_fn(dummy_output, labels)

                # create_graph=True: x에 대한 2차 미분을 가능하게 함
                # 이것이 "double backpropagation"
                dummy_gradients = torch.autograd.grad(
                    dummy_loss,
                    self.model.parameters(),
                    create_graph=True
                )

                # --- (b) 비용 함수 계산 ---
                rec_cost = self.cost_fn(dummy_gradients, target_gradients)

                # --- (c) TV 정규화 추가 ---
                if cfg['total_variation'] > 0:
                    rec_cost = rec_cost + cfg['total_variation'] * total_variation(x)

                # --- (d) x에 대한 gradient 계산 ---
                rec_cost.backward()

                # --- (e) Signed gradient 적용 ---
                # inversefed 분석:
                # Adam이 momentum을 관리하지만, x.grad를 sign()으로 대체.
                # 이는 FGSM/PGD 스타일의 adversarial attack에서 영감을 받은 것.
                # sign()은 Adam의 1st/2nd moment에만 영향을 주고,
                # 실제 업데이트는 accumulated momentum 기반이므로 수렴 가능.
                if cfg['signed']:
                    x.grad.sign_()

                optimizer.step()

                # LR 감쇠
                scheduler.step()

                # --- (f) Boxed constraint ---
                # inversefed: 정규화 공간에서의 유효 범위로 클램핑
                if cfg['boxed']:
                    with torch.no_grad():
                        x.clamp_(lower, upper)

                # --- 로깅 ---
                current_cost = rec_cost.item()
                history.append(current_cost)

                if iteration % 500 == 0 or iteration == cfg['max_iterations'] - 1:
                    print(f"  [Restart {restart_idx+1}/{cfg['restarts']}] "
                          f"Iter {iteration:5d}/{cfg['max_iterations']} | "
                          f"Cost: {current_cost:.6f}")

            # ─── 최적 결과 선택 ───
            final_cost = history[-1]
            if final_cost < best_cost:
                best_cost = final_cost
                best_result = x.detach().clone()
                best_history = history

        stats = {
            'final_cost': best_cost,
            'history': best_history,
        }

        return best_result, stats


# =============================================================================
# 6. 평가 지표 (Evaluation Metrics)
# =============================================================================
# 1번 논문 Table 5 기준: MSE, PSNR, LPIPS, SSIM

def denormalize(tensor: torch.Tensor,
                mean: torch.Tensor,
                std: torch.Tensor) -> torch.Tensor:
    """정규화된 텐서를 [0, 1] 범위로 역변환."""
    return torch.clamp(tensor.detach().clone() * std + mean, 0.0, 1.0)


def compute_mse(reconstructed: torch.Tensor,
                original: torch.Tensor,
                mean: torch.Tensor,
                std: torch.Tensor) -> float:
    """
    Mean Squared Error (↓ = 공격 성공).
    denormalize 후 [0, 1] 범위에서 계산.
    """
    rec = denormalize(reconstructed, mean, std)
    ori = denormalize(original, mean, std)
    return (rec - ori).pow(2).mean().item()


def compute_psnr(reconstructed: torch.Tensor,
                 original: torch.Tensor,
                 mean: torch.Tensor,
                 std: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio (↑ = 공격 성공).
    PSNR = 20 * log10(MAX / sqrt(MSE)), MAX=1.0
    """
    mse = compute_mse(reconstructed, original, mean, std)
    if mse == 0:
        return float('inf')
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def compute_ssim(reconstructed: torch.Tensor,
                 original: torch.Tensor,
                 mean: torch.Tensor,
                 std: torch.Tensor) -> float:
    """
    Structural Similarity Index (↑ = 공격 성공).
    간이 구현 (window_size=11, 가우시안 가중치).
    정확한 재현을 위해서는 pytorch-msssim 패키지 사용을 권장.
    """
    try:
        from pytorch_msssim import ssim as ssim_fn
        rec = denormalize(reconstructed, mean, std)
        ori = denormalize(original, mean, std)
        return ssim_fn(rec, ori, data_range=1.0, size_average=True).item()
    except ImportError:
        # Fallback: 간이 SSIM 구현
        rec = denormalize(reconstructed, mean, std)
        ori = denormalize(original, mean, std)
        return _simple_ssim(rec, ori)


def _simple_ssim(img1: torch.Tensor, img2: torch.Tensor,
                 window_size: int = 11) -> float:
    """간이 SSIM (pytorch-msssim 없을 때 사용)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1,
                             padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1,
                             padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1,
                           padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def compute_lpips(reconstructed: torch.Tensor,
                  original: torch.Tensor,
                  mean: torch.Tensor,
                  std: torch.Tensor,
                  lpips_model=None) -> float:
    """
    Learned Perceptual Image Patch Similarity (↓ = 공격 성공).
    lpips 패키지 필요: pip install lpips

    Args:
        lpips_model: 미리 로드한 lpips.LPIPS 객체 (None이면 새로 생성)
    """
    try:
        import lpips
        if lpips_model is None:
            lpips_model = lpips.LPIPS(net='alex').to(reconstructed.device)
            lpips_model.eval()

        # LPIPS는 [-1, 1] 범위 입력을 기대
        rec = denormalize(reconstructed, mean, std) * 2.0 - 1.0
        ori = denormalize(original, mean, std) * 2.0 - 1.0

        with torch.no_grad():
            score = lpips_model(rec, ori)
        return score.item()
    except ImportError:
        print("⚠️ lpips 패키지가 없습니다. pip install lpips 실행 후 다시 시도하세요.")
        return float('nan')


def compute_all_metrics(reconstructed: torch.Tensor,
                        original: torch.Tensor,
                        mean: torch.Tensor,
                        std: torch.Tensor,
                        lpips_model=None) -> Dict[str, float]:
    """
    1번 논문 Table 5 기준 4가지 지표를 한 번에 계산.

    Returns:
        {'mse': ..., 'psnr': ..., 'lpips': ..., 'ssim': ...}
    """
    metrics = {
        'mse':   compute_mse(reconstructed, original, mean, std),
        'psnr':  compute_psnr(reconstructed, original, mean, std),
        'ssim':  compute_ssim(reconstructed, original, mean, std),
        'lpips': compute_lpips(reconstructed, original, mean, std, lpips_model),
    }
    return metrics


# =============================================================================
# 7. 시각화 (Visualization)
# =============================================================================

def visualize_result(original: torch.Tensor,
                     reconstructed: torch.Tensor,
                     mean: torch.Tensor,
                     std: torch.Tensor,
                     metrics: Dict[str, float] = None,
                     save_path: str = None):
    """
    원본과 복원 이미지를 나란히 시각화.
    """
    import matplotlib.pyplot as plt

    ori = denormalize(original, mean, std)
    rec = denormalize(reconstructed, mean, std)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(ori[0].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original (Ground Truth)")
    axes[0].axis('off')

    title = "Reconstructed"
    if metrics:
        title += (f"\nMSE: {metrics['mse']:.4f} | "
                  f"PSNR: {metrics['psnr']:.2f} dB\n"
                  f"SSIM: {metrics['ssim']:.4f} | "
                  f"LPIPS: {metrics['lpips']:.4f}")
    axes[1].imshow(rec[0].permute(1, 2, 0).cpu().numpy())
    axes[1].set_title(title, fontsize=10)
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 결과 저장: {save_path}")

    plt.show()


def plot_cost_history(history: list, save_path: str = None):
    """공격 과정의 cost 변화를 시각화."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.semilogy(history)
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Cost (log scale)')
    plt.title('Attack Convergence')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


# =============================================================================
# 8. 편의 함수 (High-Level API)
# =============================================================================

def run_attack(model: torch.nn.Module,
               image: torch.Tensor,
               label: torch.Tensor,
               dataset: str = 'imagenet',
               config: dict = None,
               verbose: bool = True) -> Tuple[torch.Tensor, Dict, Dict]:
    """
    한 줄로 공격을 실행하는 편의 함수.

    사용법:
        reconstructed, metrics, stats = run_attack(model, image, label, 'cifar10')

    Args:
        model:   victim 모델
        image:   (1, C, H, W) 정규화된 입력 이미지
        label:   (1,) 정답 라벨
        dataset: 'cifar10' 또는 'imagenet'
        config:  공격 설정 (None이면 논문 기본값)
        verbose: 진행 상황 출력 여부
    Returns:
        (복원 이미지, 평가 지표, 공격 통계)
    """
    # 1. Gradient 추출 (서버가 받는 것)
    target_gradients = extract_gradients(model, image, label)
    if verbose:
        grad_norm = torch.stack([g.norm() for g in target_gradients]).mean()
        print(f"📊 Gradient norm: {grad_norm:.4e}")

    # 2. 공격 실행
    attack = GradientInversionAttack(model, dataset=dataset, config=config)
    reconstructed, stats = attack.reconstruct(
        target_gradients, label,
        num_images=image.shape[0],
        img_shape=tuple(image.shape[1:])
    )

    # 3. 평가
    metrics = compute_all_metrics(reconstructed, image, attack.dm, attack.ds)
    if verbose:
        print(f"\n📈 평가 결과:")
        print(f"   MSE:   {metrics['mse']:.6f} (↓)")
        print(f"   PSNR:  {metrics['psnr']:.2f} dB (↑)")
        print(f"   SSIM:  {metrics['ssim']:.4f} (↑)")
        print(f"   LPIPS: {metrics['lpips']:.4f} (↓)")

    return reconstructed, metrics, stats
