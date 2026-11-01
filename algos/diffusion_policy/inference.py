"""Fixed-shape FP32 CUDA graph for the Task1 global-conditioned DP policy."""
import time
import torch

class GraphPolicy:
    def __init__(self, policy, example):
        if not policy.obs_as_global_cond or policy.kwargs.get("eta", 0) != 0:
            raise ValueError("This runner requires global conditioning and deterministic DDIM")
        self.policy = policy
        self.inputs = {k: v.clone() for k, v in example.items()}
        self.noise = torch.zeros((1, policy.horizon, policy.action_dim), device=policy.device, dtype=policy.dtype)
        scheduler = policy.noise_scheduler
        scheduler.set_timesteps(policy.num_inference_steps)
        self.steps = [(int(t), torch.tensor([int(t)], device=policy.device, dtype=torch.long)) for t in scheduler.timesteps]
        started = time.perf_counter()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.output = self._forward()
        torch.cuda.synchronize()
        self.capture_s = time.perf_counter() - started

    def _forward(self):
        p = self.policy
        normalized = p.normalizer.normalize(self.inputs)
        flat = {k: x[:, :p.n_obs_steps].reshape(-1, *x.shape[2:]) for k, x in normalized.items()}
        features = p.obs_encoder(flat).reshape(1, -1)
        trajectory = self.noise
        for t, gpu_t in self.steps:
            prediction = p.model(trajectory, gpu_t, global_cond=features)
            trajectory = p.noise_scheduler.step(prediction, t, trajectory, **p.kwargs).prev_sample
        return p.normalizer["action"].unnormalize(trajectory[..., :p.action_dim])

    @torch.no_grad()
    def predict_action(self, observations, noise=None):
        for k, value in observations.items():
            self.inputs[k].copy_(value)
        if noise is None:
            self.noise.copy_(torch.randn(self.noise.shape, device=self.noise.device, dtype=self.noise.dtype))
        else:
            self.noise.copy_(noise)
        self.graph.replay()
        p = self.policy
        return {"action": self.output[:, p.n_obs_steps-1:p.n_obs_steps-1+p.n_action_steps],
                "action_pred": self.output}
