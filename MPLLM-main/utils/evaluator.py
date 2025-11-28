# utils/evaluator.py
import string
import re
import json

def evaluate_trivia(response: str, ground_truth: dict) -> dict:
    """Trivia: 檢查生成的文章中包含了多少個正確答案"""
    output_lower = response.lower()
    correct_count = 0
    answers_list = ground_truth.get("answers", [])
    
    for valid_answers in answers_list:
        for ans in valid_answers:
            if ans.lower() in output_lower:
                correct_count += 1
                break
    
    total = len(answers_list)
    score = correct_count / total if total > 0 else 0
    return {"score": score, "details": f"{correct_count}/{total}"}

def evaluate_codenames(response: str, ground_truth: dict) -> dict:
    """Codenames: 檢查生成的猜測詞是否在目標詞清單中"""
    target_words = set(w.strip().lower() for w in ground_truth.get('target_words', []))
    translator = str.maketrans('', '', string.punctuation.replace(',', '')) 
    clean_response = response.translate(translator).lower()
    predicted_words = set(w.strip() for w in clean_response.split(',') if w.strip())
    
    matched_words = predicted_words.intersection(target_words)
    matched_count = len(matched_words)
    total_targets = len(target_words)
    
    score = matched_count / total_targets if total_targets > 0 else 0
    return {"score": score, "details": f"{matched_count}/{total_targets}"}

def evaluate_logic(response: str, ground_truth: dict) -> dict:
    """Logic: 智慧評分，優先抓取 Final Answer"""
    targets_raw = ground_truth.get("targets", [])
    if not targets_raw:
        return {"score": 0, "details": "No GT"}
    
    correct_target = targets_raw[0].strip() # 正確答案，例如 "2"
    
    # === 策略 1: 抓取 "Final Answer: X" 格式 ===
    # 這能忽略中間推理過程提到的錯誤選項
    match = re.search(r'(?:Final )?Answer:\s*(\d+)', response, re.IGNORECASE)
    if match:
        prediction = match.group(1)
        if prediction == correct_target:
            return {"score": 1.0, "details": "Correct (Regex)"}
        else:
            return {"score": 0.0, "details": f"Wrong (Pred: {prediction})"}

    # === 策略 2: 嚴格排除法 (Fallback) ===
    # 如果沒有標準格式，才使用舊的邏輯
    output_lower = response.lower().strip()
    
    # 檢查是否只包含正確答案的數字，沒有其他數字
    target_aliases = {
        "1": "first", "2": "second", "3": "third", "4": "fourth", "5": "fifth",
        "6": "sixth", "7": "seventh", "8": "eighth", "9": "ninth", "10": "tenth"
    }
    
    # 建立「絕對不能出現」的錯誤選項集合
    negative_keywords = set()
    for i in range(1, 6): # 假設選項通常在 1-5 之間
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

def evaluate_response(task_type: str, response: str, ground_truth: dict) -> dict:
    if task_type == 'trivia': return evaluate_trivia(response, ground_truth)
    elif task_type == 'codenames': return evaluate_codenames(response, ground_truth)
    elif task_type == 'logic': return evaluate_logic(response, ground_truth)
    return {"score": 0, "details": "Unknown Task"}