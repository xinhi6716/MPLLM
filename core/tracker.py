# core/tracker.py
class CostTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.cost_usd = 0.0
        
        # 定價表 (每 1M tokens) - 可依需求更新
        self.PRICING = {
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
            'gpt-4o':      {'input': 5.00, 'output': 15.00},
            'default':     {'input': 0.15, 'output': 0.60} # Fallback
        }

    def add(self, model_name: str, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # 簡易計價邏輯
        key = model_name if model_name in self.PRICING else 'default'
        p = self.PRICING[key]
        
        cost = (input_tokens * p['input'] + output_tokens * p['output']) / 1_000_000
        self.cost_usd += cost

    def get_summary(self):
        return {
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "cost_usd": round(self.cost_usd, 6)
        }