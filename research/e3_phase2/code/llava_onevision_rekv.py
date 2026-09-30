import json
import os

import torch
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from logzero import logger

from model.patch import patch_hf
from model.abstract_rekv import Abstract_ReKV
from model.attention import decaf_b

# DECAF Component A instrumentation (no-op unless the env vars are set):
#   DECAF_RETRIEVAL_LOG   -> jsonl path; per-question retrieval records are appended
#   DECAF_RECALL_ONLY=1   -> stop right after retrieval (no generation); for
#                            measuring entangled-store recall without a full run
DECAF_RETRIEVAL_LOG = os.environ.get('DECAF_RETRIEVAL_LOG', '')
DECAF_RECALL_ONLY = os.environ.get('DECAF_RECALL_ONLY', '0') == '1'


def _entropy(logits):
    p = torch.softmax(logits.float(), dim=-1)
    return float(-(p * p.clamp(min=1e-9).log()).sum())


def _log_retrieval(video_id, question, input_length, kv_cache):
    if not DECAF_RETRIEVAL_LOG:
        return
    layer0 = kv_cache[0]
    rec = {
        'video_id': video_id,
        'question': question,
        'encoded_length': int(input_length),  # tokens in the store stream incl. init
        'n_blocks': int(layer0.num_global_block),
        'retrieved_layer0': [int(i) for i in layer0.retrieved_block_indices[0]],
        'retrieved_all_layers': [
            [int(i) for i in layer_kv.retrieved_block_indices[0]] for layer_kv in kv_cache
        ],
        'similarity_layer0': (
            [float(s) for s in layer0.similarity[0]] if layer0.similarity is not None else None
        ),
    }
    with open(DECAF_RETRIEVAL_LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')


class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV):
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size)

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt

    def _get_video_features(self, pixel_values_videos):
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)
        video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True)
        selected_video_feature = video_features.hidden_states[self.config.vision_feature_layer]

        if self.config.vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature
        video_features = self.multi_modal_projector(selected_video_feature)

        video_features = self.apply_pooling(video_features)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)  # (B, Nv*196, D)
        return video_features

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        def run_pass(level):
            """One retrieval+prefill+decode pass. Returns (answer, entropies, ntok, stopped)."""
            input_ids = self.processor.tokenizer(input_text['question']).input_ids
            input_ids = torch.as_tensor([input_ids], device=device)
            for layer_kv in self.kv_cache:  # activate retrieval mode
                layer_kv.set_retrieval()
            if decaf_b.DECAF_B and self.kv_cache[0].grain_store is not None:
                self.kv_cache[0].grain_store.pending_level = level
            if retrieved_indices is None:  # Internal retrieval
                out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
                past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)
                self.last_retrieved_blocks = list(self.kv_cache[0].retrieved_block_indices[0])  # T1 logging
                _log_retrieval(getattr(self, '_current_video_id', ''), input_text['question'],
                               self.kv_cache[0].length, self.kv_cache)
            else:  # External retrieval
                for layer_kv in self.kv_cache:
                    assert layer_kv.block_size == self.n_frame_tokens, f'block_size: {layer_kv.block_size}, n_frame_tokens: {self.n_frame_tokens}'
                    layer_kv.set_retrieved_block_indices(retrieved_indices)
                out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
                past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)
                self.last_retrieved_blocks = list(self.kv_cache[0].retrieved_block_indices[0])  # T1 logging
                _log_retrieval(getattr(self, '_current_video_id', ''), input_text['question'],
                               self.kv_cache[0].length, self.kv_cache)

            for layer_kv in self.kv_cache:  # reset to default
                layer_kv.reset_retrieval()

            if DECAF_RECALL_ONLY:
                return '', [], 0, True

            output_ids = []
            entropies = []
            stopped = False
            for i in range(max_new_tokens):
                if i == 0:  # prefill
                    input_ids = self.processor.tokenizer(input_text['prompt']).input_ids
                    input_ids = torch.as_tensor([input_ids], device=device)
                    inputs_embeds = self.get_input_embeddings()(input_ids)
                    out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                    past_key_values = out.past_key_values
                    logits = out.logits
                else:  # decoding
                    out = self.language_model(
                        input_ids=torch.as_tensor(
                            [[token]],
                            device=device,
                        ),
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    logits = out.logits
                    past_key_values = out.past_key_values

                last_token_logits = logits[0, -1, :]
                entropies.append(_entropy(last_token_logits))

                _, indices = torch.topk(last_token_logits, 2)
                tokens = [int(index) for index in indices.tolist()]
                token = tokens[0]

                output_ids.append(token)

                if token in stop_token_ids:
                    stopped = True
                else:
                    stopped = False

                if i == max_new_tokens - 1 or stopped:
                    break

            output = self.processor.tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            return output, entropies, len(output_ids), stopped

        answer, entropies, ntok, stopped = run_pass('coarse')

        # DECAF Component B: uncertainty-triggered commitment escalation.
        # A hedged first pass (coarse grain) is re-committed at finer grain when
        # the first-pass answer is uncertain (high token entropy or degenerate).
        commit_rec = None
        if decaf_b.DECAF_B and self.kv_cache[0].grain_store is not None:
            store = self.kv_cache[0].grain_store
            mean_ent = sum(entropies) / max(1, len(entropies))
            words = answer.split()
            degenerate = ntok <= 3 or (len(words) >= 8 and len(set(words)) <= 3)
            uncertain = (mean_ent > decaf_b.DECAF_B_ENTROPY) or degenerate
            commit_rec = {
                'video_id': getattr(self, '_current_video_id', ''),
                'question': input_text['question'],
                'pass1': dict(store.plan_log or {}, mean_entropy=round(mean_ent, 4),
                              n_tokens=ntok, answer=answer.replace('\n', '')),
                'uncertain': bool(uncertain),
            }
            if uncertain and not decaf_b.DECAF_B_NOPASS2 \
                    and decaf_b.DECAF_B_COMMIT == 'deferred':
                answer2, ent2, ntok2, _ = run_pass('fine')
                commit_rec['pass2'] = dict(store.plan_log or {},
                                           mean_entropy=round(
                                               sum(ent2) / max(1, len(ent2)), 4),
                                           n_tokens=ntok2,
                                           answer=answer2.replace('\n', ''))
                answer = answer2
            sb = store.store_bytes()
            commit_rec['store_bytes'] = sb
            decaf_b.write_commit_log(commit_rec)

        return answer


def load_model(model_path='model_zoo/LLaVA/llava-onevision-qwen2-7b-ov-hf',
               n_init=None, n_local=None, topk=64, chunk_size=1):
    device = 'cuda'
    n_frame_tokens = 196
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    
    init_prompt = '<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user '
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
    inf_llm_config = {
        'n_init': init_prompt_ids.shape[1] if n_init is None else n_init,
        'n_local': n_local,
        'fattn': False,  # V100/sm_70: triton kernel fails (LLVM slice layout error); use torch impl
        'block_size': n_frame_tokens,
        'topk': topk,
        'chunk_size': chunk_size,
        'max_cached_block': 128,
        'exc_block_size': n_frame_tokens,
        'pin_memory': True,
    }
    model = LlavaOneVision_ReKV.from_pretrained(
        model_path, 
        device_map="auto",
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        processor=processor,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        n_local=n_local,
        topk=topk,
        chunk_size=chunk_size,
    )
    model.language_model = patch_hf(model.language_model, **inf_llm_config)
    
    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()

    decaf_b.N_LAYERS = model.config.text_config.num_hidden_layers

    return model, processor
