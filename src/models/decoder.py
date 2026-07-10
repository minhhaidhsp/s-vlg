"""Language decoder: projects the fused representation into a soft prompt
prefix, conditions a causal LLM on it plus text context, and applies the
uncertainty gate at inference time.

Two backends, selected by `test_mode`:
  - test_mode=True  : tiny, non-quantized model ("sshleifer/tiny-gpt2" by
    default) — runs on local Windows/CPU, used only to validate shapes/wiring.
  - test_mode=False : the real backbone (config `llm_name`, default
    "Qwen/Qwen2.5-3B-Instruct") with QLoRA 4-bit via bitsandbytes + peft.
    bitsandbytes does not support Windows, so this path is Colab/Linux only.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class VQADecoder(nn.Module):
    # Fallback answer shown when the uncertainty gate (Eq. 32) trips at inference.
    CAUTIOUS_ANSWER = (
        "Model confidence is low for this case; recommend expert review "
        "before relying on this answer."
    )

    def __init__(
        self,
        z_final_dim: int,
        llm_name: str = "Qwen/Qwen2.5-3B-Instruct",
        test_mode: bool = True,
        tiny_model_name: str = "sshleifer/tiny-gpt2",
        n_prefix: int = 8,
        gamma: float = 1.0,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ):
        super().__init__()
        self.n_prefix = n_prefix
        self.gamma = gamma
        self.test_mode = test_mode

        model_name = tiny_model_name if test_mode else llm_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if test_mode:
            # Local Windows/CPU path: tiny model, full precision, no quantization.
            self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            # Colab/Linux path: QLoRA 4-bit base model + LoRA adapters.
            try:
                import bitsandbytes  # noqa: F401
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise RuntimeError(
                    "test_mode=False requires bitsandbytes + peft, which only work on "
                    "Linux/Colab (bitsandbytes has no Windows support). Run this on "
                    "Colab/Linux, or use test_mode=True for local shape testing on Windows."
                ) from e

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config, device_map="auto"
            )
            base_model = prepare_model_for_kbit_training(base_model)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(base_model, lora_config)

        self.d_llm = self._infer_hidden_size(self.llm.config)
        self.prefix_proj = nn.Linear(z_final_dim, n_prefix * self.d_llm)

    @staticmethod
    def _infer_hidden_size(config) -> int:
        for attr in ("hidden_size", "n_embd", "d_model"):
            if hasattr(config, attr):
                return getattr(config, attr)
        raise AttributeError("Could not infer LLM hidden size from config.")

    def project_prefix(self, z_final: torch.Tensor) -> torch.Tensor:
        """Eq. (29): P_z = Reshape(Linear(z_final)) -> [B, n_prefix, d_LLM]."""
        B = z_final.shape[0]
        return self.prefix_proj(z_final).view(B, self.n_prefix, self.d_llm)

    def _embed_text(self, text, batch_size: int, device: torch.device):
        """Tokenize text into embeddings + a real attention mask.

        `text` may be:
          - a single string, shared context (e.g. system prompt, question)
            broadcast identically to every sample in the batch; or
          - a list of `batch_size` strings, one per sample (e.g. per-patient
            Graph-RAG evidence), padded to the batch's max length.

        The empty string is valid (e.g. no evidence available yet) and simply
        contributes a zero/near-zero-length segment.
        """
        if isinstance(text, str):
            enc = self.tokenizer([text], return_tensors="pt", padding=True).to(device)
            # transformers==4.44.2's BatchEncoding.convert_to_tensors falls back to plain
            # torch.tensor(value) for an empty token list (e.g. text=""), which PyTorch
            # infers as float32 (no int elements to infer a dtype from) instead of the
            # expected int64 — breaks nn.Embedding below. Force the dtype explicitly so
            # this doesn't depend on the installed transformers version's edge-case handling.
            input_ids = enc["input_ids"].long()
            attention_mask = enc["attention_mask"].long()
            embeds = self.llm.get_input_embeddings()(input_ids)  # [1, L, d_llm]
            return (
                embeds.expand(batch_size, -1, -1),
                input_ids.expand(batch_size, -1),
                attention_mask.expand(batch_size, -1),
            )

        texts = list(text)
        assert len(texts) == batch_size, f"expected {batch_size} strings, got {len(texts)}"
        enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(device)
        input_ids = enc["input_ids"].long()
        attention_mask = enc["attention_mask"].long()
        embeds = self.llm.get_input_embeddings()(input_ids)  # [B, L, d_llm]
        return embeds, input_ids, attention_mask

    def forward(
        self,
        z_final: torch.Tensor,
        system_text: str,
        evidence_text: str,
        question_text: str,
        answer_text: str,
    ):
        """Training-time forward pass.

        Eq. (30): concatenate [P_z, E_system, E_evidence, E_question, E_answer]
        along the sequence dimension (E_answer included for teacher forcing).
        Eq. (31): the LLM consumes this sequence autoregressively.
        Eq. (33): cross-entropy loss is computed ONLY on answer-token
        positions — all prefix/system/evidence/question positions are
        labeled -100 (ignored by the loss).

        NOTE: the uncertainty gate (Eq. 32) is intentionally NOT applied here.
        The gate only fires during generate() at inference time. If training
        also suppressed answers on high-uncertainty examples, the model could
        learn the degenerate shortcut of maximizing the "needs expert review"
        refusal instead of learning to answer well; training must always
        supervise the real answer.
        """
        device = next(self.llm.parameters()).device
        B = z_final.shape[0]

        P_z = self.project_prefix(z_final)  # Eq. (29)
        prefix_mask = torch.ones(B, P_z.shape[1], dtype=torch.long, device=device)

        sys_embeds, _, sys_mask = self._embed_text(system_text, B, device)
        evid_embeds, _, evid_mask = self._embed_text(evidence_text, B, device)
        ques_embeds, _, ques_mask = self._embed_text(question_text, B, device)
        ans_embeds, ans_ids, ans_mask = self._embed_text(answer_text, B, device)

        # Eq. (30)
        inputs_embeds = torch.cat([P_z, sys_embeds, evid_embeds, ques_embeds, ans_embeds], dim=1)
        attention_mask = torch.cat([prefix_mask, sys_mask, evid_mask, ques_mask, ans_mask], dim=1)

        prefix_len = P_z.shape[1] + sys_embeds.shape[1] + evid_embeds.shape[1] + ques_embeds.shape[1]
        answer_len = ans_embeds.shape[1]
        total_len = inputs_embeds.shape[1]

        # Eq. (33): -100 everywhere except the answer-token span; padded
        # answer positions (ans_mask == 0, when answer_text is a per-sample
        # list of varying length) are also excluded from the loss.
        labels = torch.full((B, total_len), -100, dtype=torch.long, device=device)
        answer_labels = ans_ids.masked_fill(ans_mask == 0, -100)
        labels[:, prefix_len:prefix_len + answer_len] = answer_labels

        # Eq. (31): autoregressive forward; transformers computes the shifted
        # next-token cross-entropy internally from `labels`.
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return out.loss, out.logits

    @torch.no_grad()
    def generate(
        self,
        z_final: torch.Tensor,
        U: torch.Tensor,
        system_text: str,
        evidence_text: str,
        question_text: str,
        gamma: float = None,
        max_new_tokens: int = 20,
    ) -> list:
        """Inference-time generation with the uncertainty gate.

        Eq. (32): per-sample gate on the uncertainty score U (Eq. 28) —
        if U_i <= gamma, generate a normal answer; if U_i > gamma, skip
        straight to a cautious fallback answer with needs_expert_review=True.

        NOTE: this gate is applied ONLY here (inference), never in forward()/
        training — see the NOTE in forward() for why.
        """
        gamma = self.gamma if gamma is None else gamma
        device = next(self.llm.parameters()).device
        B = z_final.shape[0]

        P_z = self.project_prefix(z_final)
        prefix_mask = torch.ones(B, P_z.shape[1], dtype=torch.long, device=device)
        sys_embeds, _, sys_mask = self._embed_text(system_text, B, device)
        evid_embeds, _, evid_mask = self._embed_text(evidence_text, B, device)
        ques_embeds, _, ques_mask = self._embed_text(question_text, B, device)

        prompt_embeds = torch.cat([P_z, sys_embeds, evid_embeds, ques_embeds], dim=1)
        attention_mask = torch.cat([prefix_mask, sys_mask, evid_mask, ques_mask], dim=1)

        # TODO(efficiency): only the samples with U_i <= gamma actually need
        # generation; a production version could slice the batch and skip
        # generating for flagged samples. Kept simple (generate then override)
        # since this module is CPU/tiny-model shape-test scope for now.
        gen_ids = self.llm.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        needs_review = U > gamma
        results = []
        for i in range(B):
            if needs_review[i]:
                results.append({
                    "answer": self.CAUTIOUS_ANSWER,
                    "needs_expert_review": True,
                    "uncertainty": U[i].item(),
                })
            else:
                results.append({
                    "answer": texts[i],
                    "needs_expert_review": False,
                    "uncertainty": U[i].item(),
                })
        return results

    @torch.no_grad()
    def predict_closed(
        self,
        z_final: torch.Tensor,
        system_text: str,
        evidence_text: str,
        question_text: str,
    ) -> list:
        """Forced binary choice ("yes"/"no") for CLOSED questions.

        Free-form generation (generate()) can degenerate on a small/under-
        trained LLM: greedy decoding may collapse to the SAME output
        regardless of input (observed with the tiny test_mode backbone —
        see PROJECT_STATE.md changelog), which silently zeroes out accuracy
        on binary questions and looks like a scoring bug when it's actually
        a generation-collapse issue. For CLOSED questions, comparing the
        next-token log-probability mass assigned to "yes"-like vs "no"-like
        tokens directly is both more robust to this failure mode and the
        standard way closed/binary VQA questions are typically evaluated.

        Returns a list of "yes"/"no" strings, one per batch sample (the
        uncertainty gate is NOT applied here — callers should still check
        U/gamma themselves if a cautious fallback is desired).
        """
        device = next(self.llm.parameters()).device
        B = z_final.shape[0]

        P_z = self.project_prefix(z_final)
        prefix_mask = torch.ones(B, P_z.shape[1], dtype=torch.long, device=device)
        sys_embeds, _, sys_mask = self._embed_text(system_text, B, device)
        evid_embeds, _, evid_mask = self._embed_text(evidence_text, B, device)
        ques_embeds, _, ques_mask = self._embed_text(question_text, B, device)

        prompt_embeds = torch.cat([P_z, sys_embeds, evid_embeds, ques_embeds], dim=1)
        attention_mask = torch.cat([prefix_mask, sys_mask, evid_mask, ques_mask], dim=1)

        out = self.llm(inputs_embeds=prompt_embeds, attention_mask=attention_mask)
        next_token_log_probs = torch.log_softmax(out.logits[:, -1, :], dim=-1)  # [B, vocab]

        yes_ids = self._single_token_ids(["yes", " yes", "Yes", " Yes"])
        no_ids = self._single_token_ids(["no", " no", "No", " No"])

        yes_score = torch.logsumexp(next_token_log_probs[:, yes_ids], dim=-1)
        no_score = torch.logsumexp(next_token_log_probs[:, no_ids], dim=-1)

        return ["yes" if y > n else "no" for y, n in zip(yes_score.tolist(), no_score.tolist())]

    def _single_token_ids(self, variants: list) -> list:
        """Encodes each string in `variants` and keeps only those that map to
        exactly one token id (skips any that split into multiple sub-tokens)."""
        ids = set()
        for v in variants:
            enc = self.tokenizer.encode(v)
            if len(enc) == 1:
                ids.add(enc[0])
        if not ids:
            raise ValueError(f"None of {variants} tokenize to a single token with this tokenizer")
        return sorted(ids)


def _self_test() -> bool:
    torch.manual_seed(0)
    B = 2
    z_final_dim = 12
    n_prefix = 4
    gamma = 1.0

    decoder = VQADecoder(z_final_dim=z_final_dim, test_mode=True, n_prefix=n_prefix, gamma=gamma)
    z_final = torch.randn(B, z_final_dim)

    ok = True

    # 1) soft prefix shape — Eq. (29)
    P_z = decoder.project_prefix(z_final)
    expected_prefix_shape = (B, n_prefix, decoder.d_llm)
    if P_z.shape != expected_prefix_shape:
        print(f"FAIL: P_z shape {tuple(P_z.shape)} != {expected_prefix_shape}")
        ok = False

    # 2) training forward + loss — Eq. (30), (31), (33)
    system_text = "You are a helpful medical VQA assistant."
    evidence_text = "Graph-RAG evidence: patient has elevated creatinine."
    question_text = "What organ is likely affected?"
    answer_text = "The kidney is likely affected."

    loss, logits = decoder(z_final, system_text, evidence_text, question_text, answer_text)
    if loss.dim() != 0:
        print(f"FAIL: loss is not scalar, dim={loss.dim()}")
        ok = False
    if torch.isnan(loss):
        print("FAIL: loss is NaN")
        ok = False

    # 3) uncertainty gate — Eq. (32): one confident sample (U <= gamma), one
    #    uncertain sample (U > gamma).
    U = torch.tensor([gamma - 0.5, gamma + 0.5])
    results = decoder.generate(
        z_final, U, system_text, evidence_text, question_text, gamma=gamma, max_new_tokens=5
    )
    if results[0]["needs_expert_review"] is not False:
        print("FAIL: low-U sample incorrectly flagged for expert review")
        ok = False
    if results[1]["needs_expert_review"] is not True:
        print("FAIL: high-U sample not flagged for expert review")
        ok = False
    if results[1]["answer"] != decoder.CAUTIOUS_ANSWER:
        print("FAIL: high-U sample did not receive the cautious fallback answer")
        ok = False

    # 4) forced binary choice for CLOSED questions — must always return
    # "yes"/"no" (never degenerate free-text) and be deterministic.
    closed_preds_1 = decoder.predict_closed(z_final, system_text, evidence_text, "Is the liver normal?")
    closed_preds_2 = decoder.predict_closed(z_final, system_text, evidence_text, "Is the liver normal?")
    if len(closed_preds_1) != B or any(p not in ("yes", "no") for p in closed_preds_1):
        print(f"FAIL: predict_closed returned invalid values: {closed_preds_1}")
        ok = False
    if closed_preds_1 != closed_preds_2:
        print(f"FAIL: predict_closed is not deterministic: {closed_preds_1} != {closed_preds_2}")
        ok = False

    print("PASS: decoder" if ok else "FAIL: decoder")
    return ok


if __name__ == "__main__":
    _self_test()
