import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor


def to_pil(images: torch.Tensor) -> list[Image.Image]:
    arrays = (
        images.detach().mul(255).round().clamp(0, 255).to(torch.uint8)
        .permute(0, 2, 3, 1).cpu().numpy()
    )
    return [Image.fromarray(array) for array in arrays]


class CLIPScorer:
    def __init__(self, device):
        model_id = "openai/clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.device = device

    @torch.no_grad()
    def __call__(self, images, prompts):
        inputs = self.processor(
            text=prompts, images=to_pil(images), return_tensors="pt", padding=True
        ).to(self.device)
        output = self.model(**inputs)
        image = output.image_embeds / output.image_embeds.norm(dim=-1, keepdim=True)
        text = output.text_embeds / output.text_embeds.norm(dim=-1, keepdim=True)
        return (image * text).sum(dim=-1)


class PickScoreScorer:
    def __init__(self, device):
        self.processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, images, prompts):
        image_inputs = self.processor(
            images=to_pil(images), return_tensors="pt"
        ).to(self.device)
        text_inputs = self.processor(
            text=prompts, padding=True, truncation=True, max_length=77,
            return_tensors="pt",
        ).to(self.device)
        image = self.model.get_image_features(**image_inputs)
        text = self.model.get_text_features(**text_inputs)
        image = image / image.norm(dim=-1, keepdim=True)
        text = text / text.norm(dim=-1, keepdim=True)
        return (image * text).sum(dim=-1) * self.model.logit_scale.exp()


class HPSv2Scorer:
    def __init__(self, device):
        from hpsv2.src.open_clip import create_model, get_tokenizer

        checkpoint_path = hf_hub_download(
            repo_id="xswu/HPSv2", filename="HPS_v2.1_compressed.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = create_model(
            "ViT-H-14", precision="amp", device=device, output_dict=True
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        size = self.model.visual.image_size
        size = size[0] if isinstance(size, tuple) else size
        self.preprocess = Compose(
            [
                Resize((size, size), interpolation=InterpolationMode.BICUBIC),
                Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.tokenizer = get_tokenizer("ViT-H-14")
        self.device = device

    @torch.no_grad()
    def __call__(self, images, prompts):
        pixels = self.preprocess(images.float()).to(self.device)
        tokens = self.tokenizer(prompts).to(self.device)
        output = self.model(pixels, tokens)
        return torch.diagonal(
            output["image_features"] @ output["text_features"].T
        )


class MultiReward:
    def __init__(self, cfg, device):
        factories = {
            "pickscore": PickScoreScorer,
            "hpsv2": HPSv2Scorer,
            "clipscore": CLIPScorer,
        }
        weights = {
            "pickscore": cfg.pickscore,
            "hpsv2": cfg.hpsv2,
            "clipscore": cfg.clipscore,
        }
        self.scorers = {
            name: factories[name](device)
            for name, weight in weights.items()
            if weight != 0
        }
        self.weights = weights
        self.batch_size = cfg.batch_size

    @torch.no_grad()
    def __call__(self, images, prompts):
        details = {}
        for name, scorer in self.scorers.items():
            batches = []
            for start in range(0, len(prompts), self.batch_size):
                end = min(start + self.batch_size, len(prompts))
                batches.append(scorer(images[start:end], prompts[start:end]))
            details[name] = torch.cat(batches)
        total = sum(self.weights[name] * score for name, score in details.items())
        details["total"] = total
        return details
