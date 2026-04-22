import argparse
import contextlib
import json
import os
import gc
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def _get_model_path(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    if args.model_path:
        return args.model_path
    raise SystemExit("Model path is required via --model or --model_path.")


def _find_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    candidates = [
        ("model", "layers"),
        ("model", "h"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("decoder", "layers"),
        ("layers",),
    ]
    for path in candidates:
        obj = model
        found = True
        for name in path:
            if not hasattr(obj, name):
                found = False
                break
            obj = getattr(obj, name)
        if found and isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
            return obj
    raise RuntimeError("Unable to locate transformer layers for offload.")


def _find_norm(model: torch.nn.Module) -> torch.nn.Module | None:
    candidates = [
        ("model", "norm"),
        ("model", "final_layernorm"),
        ("transformer", "ln_f"),
        ("norm",),
    ]
    for path in candidates:
        obj = model
        found = True
        for name in path:
            if not hasattr(obj, name):
                found = False
                break
            obj = getattr(obj, name)
        if found and isinstance(obj, torch.nn.Module):
            return obj
    return None


class LayerOffloader:
    def __init__(self, layers: torch.nn.ModuleList, device: torch.device) -> None:
        self.layers = layers
        self.device = device
        self.count = len(layers)
        self.prefetch_stream = torch.cuda.Stream()
        self.offload_stream = torch.cuda.Stream()
        self.prefetch_events = [torch.cuda.Event() for _ in range(self.count)]
        self.prefetched = [False] * self.count
        self.prefetch_depth = int(os.environ.get("PREFILL_PREFETCH_DEPTH", "2"))
        self.keep_on_gpu = int(os.environ.get("PREFILL_KEEP_LAYERS", "1"))
        self._hooks = []
        self._pin_layers()
        for idx, layer in enumerate(layers):
            self._hooks.append(layer.register_forward_pre_hook(self._make_pre_hook(idx)))
            self._hooks.append(layer.register_forward_hook(self._make_post_hook(idx)))

    def _pin_layers(self) -> None:
        for layer in self.layers:
            for param in layer.parameters(recurse=False):
                if not param.is_cuda:
                    param.data = param.data.pin_memory()
            for buf in layer.buffers(recurse=False):
                if not buf.is_cuda:
                    buf.data = buf.data.pin_memory()

    @staticmethod
    def _is_on_gpu(layer: torch.nn.Module) -> bool:
        for param in layer.parameters(recurse=False):
            return param.is_cuda
        for buf in layer.buffers(recurse=False):
            return buf.is_cuda
        return False

    def _make_pre_hook(self, idx: int):
        def hook(module, inputs):
            if self.prefetched[idx]:
                torch.cuda.current_stream().wait_event(self.prefetch_events[idx])
                self.prefetched[idx] = False
            else:
                module.to(self.device, non_blocking=True)
            if self.prefetch_depth > 0:
                last_idx = min(self.count, idx + 1 + self.prefetch_depth)
                for next_idx in range(idx + 1, last_idx):
                    if self.prefetched[next_idx] or self._is_on_gpu(self.layers[next_idx]):
                        continue
                    with torch.cuda.stream(self.prefetch_stream):
                        self.layers[next_idx].to(self.device, non_blocking=True)
                        self.prefetch_events[next_idx].record(self.prefetch_stream)
                    self.prefetched[next_idx] = True
            return None

        return hook

    def _make_post_hook(self, idx: int):
        def hook(module, inputs, outputs):
            offload_idx = idx - self.keep_on_gpu
            if offload_idx >= 0:
                done_event = torch.cuda.Event()
                done_event.record(torch.cuda.current_stream())
                with torch.cuda.stream(self.offload_stream):
                    self.offload_stream.wait_event(done_event)
                    self.layers[offload_idx].to("cpu", non_blocking=True)
            return outputs

        return hook

    def prefetch_first(self) -> None:
        if self.count == 0:
            return
        with torch.cuda.stream(self.prefetch_stream):
            self.layers[0].to(self.device, non_blocking=True)
            self.prefetch_events[0].record(self.prefetch_stream)
        self.prefetched[0] = True

    def offload_all(self) -> None:
        with torch.cuda.stream(self.offload_stream):
            for layer in self.layers:
                layer.to("cpu", non_blocking=True)
        torch.cuda.current_stream().wait_stream(self.offload_stream)


def _compute_loss_per_sample(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = attention_mask[:, 1:].to(torch.float32)
    vocab = shift_logits.size(-1)
    loss = F.cross_entropy(
        shift_logits.reshape(-1, vocab),
        shift_labels.reshape(-1),
        reduction="none",
    )
    loss = loss.view(shift_labels.size())
    denom = shift_mask.sum(dim=1).clamp(min=1)
    return (loss * shift_mask).sum(dim=1) / denom


def _compute_loss_per_sample_chunked(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    lm_head: torch.nn.Module,
    max_logit_tokens: int,
) -> torch.Tensor:
    shift_hidden = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = attention_mask[:, 1:].to(torch.float32)
    batch = shift_hidden.size(0)
    seq_len = shift_hidden.size(1)
    loss_sum = torch.zeros(batch, device=shift_hidden.device, dtype=torch.float32)
    token_count = torch.zeros(batch, device=shift_hidden.device, dtype=torch.float32)
    chunk_len = max(1, max_logit_tokens // batch)
    vocab = None
    for start in range(0, seq_len, chunk_len):
        end = min(seq_len, start + chunk_len)
        hs_chunk = shift_hidden[:, start:end, :]
        logits = lm_head(hs_chunk)
        if vocab is None:
            vocab = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab),
            shift_labels[:, start:end].reshape(-1),
            reduction="none",
        )
        loss = loss.view(batch, -1)
        mask_chunk = shift_mask[:, start:end]
        loss_sum += loss * mask_chunk
        token_count += mask_chunk
    return loss_sum / token_count.clamp(min=1)


def _get_base_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "base_model") and model.base_model is not None:
        return model.base_model
    prefix = getattr(model, "base_model_prefix", None)
    if prefix and hasattr(model, prefix):
        return getattr(model, prefix)
    if hasattr(model, "model"):
        return getattr(model, "model")
    return model


def _build_batches_by_tokens(
    sorted_indices: list[int], lengths: list[int], max_tokens: int
) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    max_len = 0
    for idx in sorted_indices:
        length = lengths[idx]
        next_max = max(max_len, length)
        next_tokens = next_max * (len(current) + 1)
        if current and next_tokens > max_tokens:
            batches.append(current)
            current = []
            max_len = 0
        current.append(idx)
        max_len = max(max_len, length)
    if current:
        batches.append(current)
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefiller")
    parser.add_argument("--model", help="Model Path")
    parser.add_argument("--model_path", help="Model Path (alias for --model)")
    args = parser.parse_args()
    model_path = _get_model_path(args)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    model.config.use_cache = False

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    offloader = None
    if use_cuda:
        layers = _find_layers(model)
        inp_emb = model.get_input_embeddings()
        if inp_emb is not None:
            inp_emb.to(device)
        out_emb = model.get_output_embeddings()
        if out_emb is not None:
            out_emb.to(device)
        norm = _find_norm(model)
        if norm is not None:
            norm.to(device)
        offloader = LayerOffloader(layers, device)
    else:
        model.to(device)

    print("ready", flush=True)

    raw = input()
    texts = json.loads(raw)
    lengths = [len(tokenizer(t).input_ids) for t in texts]
    sorted_indices = sorted(range(len(texts)), key=lengths.__getitem__, reverse=True)
    max_tokens = int(os.environ.get("PREFILL_MAX_TOKENS", "49152"))
    max_logit_tokens = int(os.environ.get("PREFILL_LOGIT_TOKENS", "4096"))
    batches = _build_batches_by_tokens(sorted_indices, lengths, max_tokens=max_tokens)
    results: list[float] = [0.0] * len(texts)
    base_model = _get_base_model(model)
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Failed to locate lm_head for loss computation.")

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_cuda
        else contextlib.nullcontext()
    )
    enable_compile = False
    if use_cuda and hasattr(torch, "compile"):
        compile_flag = os.environ.get("PREFILL_COMPILE", "0")
        if compile_flag not in {"0", "false", "False"}:
            enable_compile = True
            if offloader is not None:
                force_flag = os.environ.get("PREFILL_COMPILE_FORCE", "0")
                if force_flag in {"0", "false", "False"}:
                    enable_compile = False
    if enable_compile:
        try:
            import torch._dynamo as dynamo

            dynamo.config.suppress_errors = True
        except Exception:
            dynamo = None
        compile_mode = os.environ.get("PREFILL_COMPILE_MODE", "reduce-overhead")
        try:
            base_model = torch.compile(
                base_model, mode=compile_mode, fullgraph=False, dynamic=True
            )
        except Exception:
            pass

    def run_batch(batch_indices: list[int]) -> None:
        encoded = None
        input_ids = None
        attention_mask = None
        outputs = None
        loss = None
        try:
            batch_texts = [texts[i] for i in batch_indices]
            encoded = tokenizer(batch_texts, return_tensors="pt", padding=True)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(device)
            if offloader is not None:
                offloader.prefetch_first()
            with autocast_ctx:
                outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            hidden_states = outputs[0]
            outputs = None
            loss = _compute_loss_per_sample_chunked(
                hidden_states, input_ids, attention_mask, lm_head, max_logit_tokens
            )
            hidden_states = None
            for idx, value in zip(batch_indices, loss.detach().cpu().tolist()):
                results[idx] = float(value)
        finally:
            encoded = None
            input_ids = None
            attention_mask = None
            outputs = None
            loss = None
            if use_cuda:
                torch.cuda.synchronize()

    def process_batch(batch_indices: list[int]) -> None:
        try:
            run_batch(batch_indices)
        except torch.cuda.OutOfMemoryError:
            if len(batch_indices) == 1:
                raise
            if offloader is not None:
                offloader.offload_all()
            if use_cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            mid = len(batch_indices) // 2
            process_batch(batch_indices[:mid])
            process_batch(batch_indices[mid:])

    with torch.inference_mode():
        for batch_indices in batches:
            process_batch(batch_indices)

    for value in results:
        print(value, flush=True)


if __name__ == "__main__":
    main()
