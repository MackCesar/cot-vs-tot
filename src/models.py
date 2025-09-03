import os
from dataclasses import dataclass
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Simple config holding a backend-prefixed model string.
    Examples:
      - "openai:gpt-4o-mini"
      - "hf:meta-llama/Llama-3.1-8B-Instruct"
      - "ollama:llama3"
    """
    model: str


class ModelClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.kind, self.name = self._parse(cfg.model)
        self._client = None
        self._pipe = None
        self._ollama = None

        if self.kind == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.kind == "hf":
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            tok = AutoTokenizer.from_pretrained(self.name)
            mdl = AutoModelForCausalLM.from_pretrained(self.name)
            self._pipe = pipeline(
                "text-generation",
                model=mdl,
                tokenizer=tok,
                device_map="auto",
            )
        elif self.kind == "ollama":
            import ollama
            self._ollama = ollama
        else:
            raise ValueError(f"Unknown model kind: {self.kind}")

    def _parse(self, s: str) -> Tuple[str, str]:
        """Parse strings like "openai:gpt-4o-mini" -> ("openai", "gpt-4o-mini").
        Defaults to HuggingFace if no prefix is provided.
        """
        if ":" in s:
            k, n = s.split(":", 1)
            return k.strip(), n.strip()
        return "hf", s.strip()

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        if self.kind == "openai":
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            resp = self._client.chat.completions.create(
                model=self.name,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content

        elif self.kind == "hf":
            full_prompt = (system + "\n\n" if system else "") + prompt
            out = self._pipe(
                full_prompt,
                do_sample=temperature > 0,
                temperature=max(0.1, temperature),
                max_new_tokens=max_tokens,
            )
            text = out[0]["generated_text"]
            # Some pipelines return the prompt + completion; strip the prompt prefix if present
            if text.startswith(full_prompt):
                text = text[len(full_prompt):]
            return text.strip()

        elif self.kind == "ollama":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = self._ollama.chat(model=self.name, messages=messages)
            return resp["message"]["content"]

        else:
            raise RuntimeError("Unsupported backend kind")