"""논문 알고리즘 2의 완화된 MCMC 경계 추출과 고정 크기 데이터 교체를 구현한다."""

import numpy as np
import torch


def boundary_samples(model, limits, count, rng, *, sigma_e=.1, proposal_std=.05, u_min=.8, steps=1000, chunk=2048, device="cpu", initial_q=None):
    r"""모델의 영거리 경계 주변에서 독립적인 병렬 랜덤워크의 마지막 상태를 뽑는다.

    $$h(q)=\exp(-\hat d(q)^2/\sigma_e^2),\qquad q^+=q+\epsilon,\quad\epsilon\sim\mathcal N(0,s^2I)$$

    sigma_e는 거리 척도(m), s는 proposal_std(rad)다. 논문의 sigma를 제안 표준편차로 해석한다.
    u는 논문대로 u_min부터 1 사이에서 뽑으므로 표준 MH의 정확한 정상분포를 보장하지 않는다.
    """
    if count <= 0 or steps <= 0 or chunk <= 0 or not 0 <= u_min < 1:
        raise ValueError("MCMC 표본 수·횟수·묶음 크기와 수용 범위를 확인하세요.")
    if not np.isfinite([sigma_e, proposal_std]).all() or sigma_e <= 0 or proposal_std <= 0:
        raise ValueError("MCMC 거리 척도와 제안 표준편차는 유한한 양수여야 합니다.")
    if initial_q is not None:
        initial_q = np.asarray(initial_q, dtype=np.float32)
        if (initial_q.shape != (count, len(limits)) or not np.isfinite(initial_q).all()
                or np.any(initial_q < limits[:, 0]) or np.any(initial_q > limits[:, 1])):
            raise ValueError("MCMC 시작 자세의 크기·유한성·관절 범위를 확인하세요.")
    if device == "cuda":
        return boundary_samples_cuda(model, limits, count, rng, sigma_e, proposal_std, u_min, steps, initial_q)
    if device != "cpu":
        raise ValueError("지원하지 않는 MCMC 장치입니다.")
    model.eval()
    samples = []
    accepted = 0
    with torch.inference_mode():
        for begin in range(0, count, chunk):
            size = min(chunk, count - begin)
            q = (rng.uniform(limits[:, 0], limits[:, 1], size=(size, len(limits))).astype(np.float32)
                 if initial_q is None else initial_q[begin:begin + size].copy())
            distance = model(torch.from_numpy(q)).cpu().numpy()
            # 지수의 언더플로를 피하는 목표 로그밀도: $$\ell(q)=-\hat d(q)^2/\sigma_e^2$$
            log_density = -(distance / sigma_e) ** 2
            for _ in range(steps):
                # 대칭 정규 제안으로 관절각을 이동: $$q^+=q+s z,\quad z\sim\mathcal N(0,I)$$
                proposed = q + rng.normal(0, proposal_std, q.shape).astype(np.float32)
                valid = np.all((proposed >= limits[:, 0]) & (proposed <= limits[:, 1]), axis=1)
                evaluated = np.where(valid[:, None], proposed, q)
                distance = model(torch.from_numpy(evaluated)).cpu().numpy()
                # 새 제안의 목표 로그밀도: $$\ell(q^+)=-\hat d(q^+)^2/\sigma_e^2$$
                next_density = -(distance / sigma_e) ** 2
                log_u = np.log(rng.uniform(max(u_min, np.finfo(float).tiny), 1, size))
                # 논문 알고리즘 2의 수용 조건: $$\ell(q^+)-\ell(q)>\log u$$
                accept = valid & (next_density - log_density > log_u)
                q[accept] = proposed[accept]
                log_density[accept] = next_density[accept]
                accepted += int(accept.sum())
            samples.append(q)
    return np.concatenate(samples), {"acceptance_fraction": accepted / (count * steps)}


def boundary_samples_cuda(model, limits, count, rng, sigma_e, proposal_std, u_min, steps, initial_q=None):
    r"""CPU와 같은 경계 추출 규칙을 CUDA 위의 병렬 체인으로 실행한다.

    $$\log h(q^+)-\log h(q)>\log u,\qquad u\sim U(u_{min},1)$$

    난수 생성과 수용 상태를 GPU에 유지하고 최종 관절각만 CPU로 가져온다.
    """
    model.eval()
    generator = torch.Generator(device="cuda").manual_seed(int(rng.integers(0, 2**31)))
    lower = torch.as_tensor(limits[:, 0], dtype=torch.float32, device="cuda")
    upper = torch.as_tensor(limits[:, 1], dtype=torch.float32, device="cuda")
    accepted = torch.zeros((), dtype=torch.int64, device="cuda")
    samples = []
    with torch.no_grad():
        for begin in range(0, count, 100_000):
            size = min(100_000, count - begin)
            if initial_q is None:
                unit = torch.rand((size, len(limits)), generator=generator, device="cuda")
                # 관절 제한 안의 균일 초기 표본: $$q=q_{min}+u\odot(q_{max}-q_{min})$$
                q = lower + unit * (upper - lower)
            else:
                q = torch.as_tensor(initial_q[begin:begin + size], device="cuda").clone()
            # 목표 로그밀도: $$\ell(q)=-\hat d(q)^2/\sigma_e^2$$
            density = -(model(q) / sigma_e).square()
            if not torch.isfinite(density).all():
                raise RuntimeError("CUDA MCMC의 초기 거리 예측이 유한하지 않습니다.")
            for _ in range(steps):
                noise = torch.randn(q.shape, generator=generator, device="cuda")
                # 대칭 정규 제안: $$q^+=q+s z$$
                proposed = q + proposal_std * noise
                valid = torch.all((proposed >= lower) & (proposed <= upper), dim=1)
                evaluated = torch.where(valid[:, None], proposed, q)
                # 새 제안의 목표 로그밀도: $$\ell(q^+)=-\hat d(q^+)^2/\sigma_e^2$$
                next_density = -(model(evaluated) / sigma_e).square()
                uniform = torch.rand((size,), generator=generator, device="cuda")
                # 완화된 수용 판단에 사용할 로그 난수: $$\log u=\log(u_{min}+(1-u_{min})v)$$
                log_u = torch.log((u_min + (1 - u_min) * uniform).clamp_min(torch.finfo(torch.float32).tiny))
                # 관절 제한과 논문 알고리즘 2의 수용 조건: $$a=\mathbf1_{valid}\mathbf1_{\ell(q^+)-\ell(q)>\log u}$$
                accept = valid & torch.isfinite(next_density) & (next_density - density > log_u)
                q = torch.where(accept[:, None], proposed, q)
                density = torch.where(accept, next_density, density)
                accepted += accept.sum()
            samples.append(q.cpu().numpy())
    return np.concatenate(samples), {"acceptance_fraction": int(accepted) / (count * steps), "device": "cuda"}


def replace_random(q, distances, new_q, new_distances, rng):
    """논문 알고리즘 1처럼 기존 자료를 무작위로 골라 같은 개수의 새 정답으로 교체한다."""
    if len(new_q) != len(new_distances) or not 0 < len(new_q) <= len(q):
        raise ValueError("교체할 자료 크기가 올바르지 않습니다.")
    indices = rng.choice(len(q), size=len(new_q), replace=False)
    q[indices] = new_q
    distances[indices] = new_distances
    return indices
