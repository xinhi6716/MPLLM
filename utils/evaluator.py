# utils/evaluator.py
import string
import re
import json

# === Helper Functions ===
def normalize_answer(ans):
    """Trivia 專用：移除停用詞、標點符號的正規化工具"""
    if not ans:
        return ""
    s = str(ans).lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(the|a|an|and|or|of|in|on|at|to|for|with|by)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_codenames_response(response) -> set:
    """Codenames 專用解析器"""
    answer_text = str(response).strip()

    # 1. 嘗試解析 JSON 或處理 Dict
    if isinstance(response, dict):
        val = response.get("guessed_words") or response.get("guess") or response.get("words") or list(response.values())[0]
        answer_text = str(val)
    else:
        try:
            data = json.loads(answer_text)
            if isinstance(data, list):
                answer_text = ",".join(map(str, data))
            elif isinstance(data, dict):
                val = data.get("guessed_words") or data.get("guess") or data.get("words") or list(data.values())[0]
                answer_text = str(val)
        except:
            pass

    # 2. 清理與分割
    answer_text = re.sub(r'^(?:Answer|Final answer|Guesses|Guess|Output)[:\s]*', '', answer_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"[^a-zA-Z0-9]+", ",", answer_text)
    predicted_words = set(w.strip().lower() for w in clean_text.split(',') if w.strip())
    
    return predicted_words

# === Main Evaluation Functions ===

def evaluate_trivia(response, ground_truth: dict) -> dict:
    """Trivia: 修正了 dict 物件處理問題"""
    
    # [關鍵修正] 這裡處理傳入的是 Dict 的情況
    if isinstance(response, dict):
        # 優先抓取 story，沒有的話抓 answers 列表
        text_content = response.get("final_story") or str(response.get("final_answers")) or str(response)
    else:
        text_content = str(response)
        # 嘗試解析 JSON 字串
        try:
            data = json.loads(text_content)
            if isinstance(data, dict):
                text_content = data.get("final_story") or str(data.get("final_answers")) or str(data)
        except:
            pass

    # 正規化內容
    final_norm = normalize_answer(text_content)
    correct_count = 0
    answers_list = ground_truth.get("answers", [])
    
    for valid_answers in answers_list:
        hit = False
        for correct_ans in valid_answers:
            correct_norm = normalize_answer(correct_ans)
            if (final_norm == correct_norm 
                or final_norm in correct_norm 
                or correct_norm in final_norm
                or any(w in correct_norm for w in [w for w in final_norm.split() if len(w) > 2])
            ):
                hit = True
                break
        if hit:
            correct_count += 1
    
    total = len(answers_list)
    score = correct_count / total if total > 0 else 0
    return {"score": score, "details": f"{correct_count}/{total}"}

def evaluate_codenames(response, ground_truth: dict) -> dict:
    target_words = set(w.strip().lower() for w in ground_truth.get('target_words', []))
    predicted_words = parse_codenames_response(response)
    
    matched_words = predicted_words.intersection(target_words)
    matched_count = len(matched_words)
    total_targets = len(target_words)
    
    score = matched_count / total_targets if total_targets > 0 else 0.0
    matched_str = ", ".join(list(matched_words)[:3])
    if len(matched_words) > 3: matched_str += "..."
    
    return {"score": score, "details": f"{matched_count}/{total_targets} ({matched_str})"}

def evaluate_logic(response, ground_truth: dict) -> dict:
    targets_raw = ground_truth.get("targets", [])
    if not targets_raw:
        return {"score": 0, "details": "No GT"}
    
    correct_target = str(targets_raw[0]).strip()
    
    # 處理 Dict
    if isinstance(response, dict):
        response_str = str(response.get('final_answer', response))
    else:
        response_str = str(response)
        try:
             data = json.loads(response_str)
             if isinstance(data, dict):
                 response_str = str(data.get('final_answer', data))
        except:
             pass

    # Regex 策略
    match = re.search(r'(?:Final )?Answer:\s*(\d+)', response_str, re.IGNORECASE)
    if match:
        prediction = match.group(1)
        if prediction == correct_target:
            return {"score": 1.0, "details": "Correct (Regex)"}
        else:
            return {"score": 0.0, "details": f"Wrong (Pred: {prediction})"}

    # Fallback 策略
    output_lower = response_str.lower().strip()
    target_aliases = {"1": "first", "2": "second", "3": "third", "4": "fourth", "5": "fifth"}
    negative_keywords = set()
    for i in range(1, 6): 
        s_i = str(i)
        if s_i != correct_target:
            negative_keywords.add(s_i)
            if s_i in target_aliases:
                negative_keywords.add(target_aliases[s_i])

    hit_negative = any(n in output_lower for n in negative_keywords)
    hit_positive = correct_target in output_lower
    
    if hit_positive and not hit_negative:
        return {"score": 1.0, "details": "Correct (Strict)"}
    
    return {"score": 0.0, "details": "Wrong (Strict)"}

def evaluate_response(task_type: str, response, ground_truth: dict) -> dict:
    if task_type == 'trivia': return evaluate_trivia(response, ground_truth)
    elif task_type == 'codenames': return evaluate_codenames(response, ground_truth)
    elif task_type == 'logic': return evaluate_logic(response, ground_truth)
    return {"score": 0, "details": "Unknown Task"}