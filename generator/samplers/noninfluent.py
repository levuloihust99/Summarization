import torch


def noninfluent_sampler(
    logits: torch.Tensor, topk_min: int = 1, topk_max: int = 3, num_consecutive: int = 3
):
    if not hasattr(noninfluent_sampler, "state"):
        setattr(noninfluent_sampler, "state", {"count": 1})
    count = noninfluent_sampler.state["count"]
    noninfluent_sampler.state["count"] += 1
    if count % (num_consecutive + 1) == 0:
        next_ids = torch.argmax(logits, dim=-1)
    else:
        _, next_probable_ids = torch.topk(logits, k=topk_max, dim=-1)
        bsz = next_probable_ids.size(0)
        selected = torch.multinomial(
            input=torch.tile(
                torch.tensor(
                    [0] * (topk_min - 1) + [1] * (topk_max - topk_min + 1),
                    dtype=torch.float,
                ).to(logits.device),
                dims=(bsz, 1),
            ),
            num_samples=1,
        )  # categorical distribution, [bsz, 1]
        next_ids = torch.gather(next_probable_ids, dim=-1, index=selected.unsqueeze(-1))
        next_ids = next_ids.squeeze(-1)
    return next_ids
