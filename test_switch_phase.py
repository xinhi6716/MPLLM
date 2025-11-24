import os
import json
import re
from typing import Dict, Any, Tuple

from mpllm_prompts.nano import nano_run
from mpllm_prompts.switch_prompts import trivia_personas_switch_prompt

# API 設定
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5-mini")


def run_switch_phase(model_fn, topic: str, verbose: bool = False) -> Tuple[Dict[str, Any], int]:
    """測試用的 run_switch_phase 函數"""
    switch_prompt = trivia_personas_switch_prompt.replace("{topic}", topic)

    def limited_switch_model_fn(messages):
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_completion_tokens=900,
            # response_format={"type":"json_object"},  # 若可用就打開
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        total_tokens = int(getattr(usage, "total_tokens", 0) if usage else 0)
        return text, total_tokens

    switch_result, switch_tokens, _ = nano_run(
        persona="", user_text=switch_prompt, model_fn=limited_switch_model_fn
    )

    try:
        personas_data = json.loads(switch_result)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", switch_result, flags=re.S)
        if m:
            try:
                personas_data = json.loads(m.group(0))
            except json.JSONDecodeError:
                personas_data = {"groups":[
                    {"g":1,"r":"Fact Expert","t":"Story Weaver"},
                    {"g":2,"r":"Data Scholar","t":"Tale Scribe"},
                    {"g":3,"r":"Knowledge Keeper","t":"Myth Builder"},
                ]}
        else:
            personas_data = {"groups":[
                {"g":1,"r":"Fact Expert","t":"Story Weaver"},
                {"g":2,"r":"Data Scholar","t":"Tale Scribe"},
                {"g":3,"r":"Knowledge Keeper","t":"Myth Builder"},
            ]}

    if verbose:
        print(f"步驟 1: 生成角色群組 OK ({switch_tokens} tokens)")
        for g in personas_data.get("groups", []):
            print(f"   群組{g.get('g')}: {g.get('r')} | {g.get('t')}")

    return personas_data, switch_tokens


def test_trivia():
    """測試 run_switch_phase 函數"""
    # 測試主題
    topic = "Pop Culture"
    
    # model_fn 參數在此函數中不需要，因為 limited_switch_model_fn 是在內部定義的
    # 但為了保持接口一致性，我們傳入 None
    personas_data, switch_tokens = run_switch_phase(model_fn=None, topic=topic, verbose=True)
    
    print(f"\n=== 測試結果 ===")
    print(f"消耗 Tokens: {switch_tokens}")
    print(f"生成的角色群組:")
    print(json.dumps(personas_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_trivia()

