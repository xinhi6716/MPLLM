MPLLM Nano 函式建置日誌
**日期**: 2025年09月17日  
**專案**: MPLLM 依賴注入 nano() 函式建置與測試

# 任務概述
建置一個 nano() 函式，實現依賴注入模式，負責組合 persona + prompt，實際模型呼叫由外部傳入的 model_fn 決定。

# 實作過程

1. 初始需求分析
**目標**: 建立 `nano()` 函式，實現依賴注入
**功能**: 只負責組合 persona + prompt，模型呼叫由外部決定
**測試**: 使用 gpt-4.1-mini 驗證功能

2. 核心函式實作 (`mpllm_prompts/nano.py`)

2.1 新增 nano() 函式
```python
def nano(persona: str, prompt: str, model_fn: ModelFn) -> Tuple[str, int, List[Dict[str, str]]]:
    """依賴注入最小實作：
    - nano() 只負責把 persona + prompt 組成 messages
    - 實際呼叫哪個模型由外部傳入的 model_fn 決定
    
    回傳 (out_text, tokens, messages)
    """
    messages = nano_build(persona, prompt)
    out_text, tokens = model_fn(messages)
    return out_text, tokens, messages
```

2.2 現有函式保持不變
- `nano_build()`: 組合 persona 與使用者文字，回傳 messages
- `nano_run()`: 依賴注入版本，內部使用 `nano_build()`

3. 測試程式建置 (`test/test.py`)

3.1 資料讀取功能
- 實作 `load_trivia_data()`, `load_codenames_data()`, `load_logic_data()`
- 支援從 JSONL 檔案讀取測試資料

3.2 Prompt 整合測試

4. 資料格式適配

4.1 JSONL 資料結構分析
```json
{
  "questions": ["問題1", "問題2", "問題3", "問題4", "問題5"],
  "answers": [["答案1"], ["答案2"], ...],
  "question_ids": ["tc_2", "tc_33", ...],
  "topic": "主題名稱"
}
```

4.2 Prompt 參數對應
`{n}`: 問題數量 (len(questions))
`{topic}`: 主題名稱
`{questions}`: 問題列表字串

5. 測試功能完善

5.1 隨機題目選擇
- 從 trivia_creative_writing_100_n_5.jsonl 隨機選取題目組
- 顯示完整的題目資訊和索引

5.2 詳細輸出格式
```
=== 選中的題目組 (第 X 組) ===
主題: [主題名稱]
問題數量: [數量]
問題列表:
  1. [問題1]
  2. [問題2]
  ...
資料筆數: [總數]
選中索引: [索引]

=== 生成的人格群組 ===
[JSON 格式的人格群組]
=== Tokens === [token數量]
```

6. 遇到的問題與解決方案

6.1 JSON 格式衝突
**問題**: `KeyError: '\n  "groups"'`
**原因**: prompt 中的 JSON 範例花括號與 `.format()` 衝突
**解決**: 使用 `.replace()` 而非 `.format()`

6.3 資料路徑錯誤
**問題**: 檔案路徑不正確
**解決**: 修正為 `./data/trivia_creative_writing/trivia_creative_writing_100_n_5.jsonl`

7. 最終成果

7.1 核心功能
✅ `nano()` 依賴注入函式完成  
✅ `nano_build()` 訊息組裝功能  
✅ `nano_run()` 完整執行流程  

7.2 測試系統
✅ 隨機題目選擇  
✅ OpenAI API 整合  
✅ 詳細輸出顯示  
✅ 錯誤處理機制  

7.3 資料整合
✅ JSONL 資料讀取  
✅ Prompt 格式適配  
✅ 多問題批次處理  

8. Codenames 測試功能擴展

8.1 新增 `test_codenames()` 函式
- **對照 SPP 實作**: 參考原有的 `codenames_single_test()` 邏輯
- **資料格式**: 使用 `target_words` 和 `word_list` 欄位
- **Prompt 整合**: 使用 `codenames_personas_switch_prompt`
- **簡化流程**: 只輸出 Spymaster 提示詞 + 人格群組，無 Guesser 流程

8.2 Codenames 資料結構
```json
{
  "word_list": ["候選單字1", "候選單字2", ...],
  "target_words": ["目標單字1", "目標單字2", ...],
  "idx": 題目編號
}
```

8.3 Codenames Prompt 參數
- `{n}`: 目標單字數量
- `{target_words}`: 目標單字列表
- `{word_list}`: 候選單字清單

9. Logic Grid Puzzle 測試功能擴展

9.1 新增 `test_logic()` 函式
- **對照 SPP 實作**: 參考原有的 `logic_single_test()` 邏輯 (200-223行)
- **資料格式**: 使用 `inputs` 和 `targets` 欄位
- **Prompt 整合**: 使用 `logic_personas_switch_prompt`
- **簡化流程**: 只生成人格群組，無推理過程

9.2 Logic Grid Puzzle 資料結構
```json
{
  "inputs": "邏輯題目文字描述",
  "targets": ["標準答案1", "標準答案2", ...]
}
```

9.3 Logic Grid Puzzle Prompt 參數
- `{input}`: 邏輯題目的完整文字描述

10. 命令列介面擴展

10.1 支援三種任務類型
```bash
python test.py trivia    # 測試 Trivia 人格生成
python test.py codename  # 測試 Codenames Spymaster 人格生成
python test.py logic     # 測試 Logic Grid Puzzle 人格生成
```

10.2 互動式選擇介面
- 不帶參數執行時提供選單
- 支援數字選項 (1, 2, 3)
- 支援文字選項 (trivia, codename, logic)

11. 輸出格式統一化

11.1 所有測試的統一輸出結構
```
=== 選中的 [任務類型] 題目 (第 X 題) ===
[題目相關資訊]
資料筆數: [總數]
選中索引: [索引]

=== 生成的 [任務類型] 人格群組 ===
[JSON 格式的人格群組]
=== Tokens === [token數量]
```

11.2 任務特定說明
- **Trivia**: 顯示主題、問題數量、問題列表
- **Codenames**: 顯示目標單字、候選清單，標註"只輸出 Spymaster 部分"
- **Logic**: 顯示完整題目文字、標準答案，標註"只生成人格，無推理過程"

12. Decider 人格設計規則

12.1 Trivia Decider 規則
- **角色定位**: 一個全域的決策者，整合所有 researcher 和 thinker 的意見
- **功能**: 產生符合任務要求的最終答案
- **特點**: 必須是中性的整合者/判斷者
- **範例**: "Judge", "Referee", "Editorial Board Chair"
- **要求**: 單一 decider，所有群組共享

12.2 Codenames Decider 規則
- **角色定位**: 團隊領導者（非 Spymaster），整合 researcher/thinker 輸出
- **功能**: 在後續步驟中整合所有意見
- **特點**: 必須與 Spymaster 角色分離
- **範例**: "Team Lead", "Strategy Coordinator", "Decision Maker"
- **要求**: 單一全域 decider，負責最終整合

12.3 Logic Grid Puzzle Decider 規則
- **角色定位**: 邏輯推理的仲裁者，驗證和整合推理結果
- **功能**: 檢查邏輯一致性，做出最終決定
- **特點**: 必須具備邏輯分析能力
- **範例**: "Logic Judge", "Reasoning Arbiter", "Deduction Validator"
- **要求**: 單一 decider，負責邏輯驗證和最終決策

12.4 共同 Decider 設計原則
- **數量限制**: 所有任務類型都只有一個全域 decider
- **中性特質**: 必須保持客觀中立的立場
- **整合功能**: 負責整合多個群組的輸出
- **最終決策**: 擁有最終決策權，產生最終答案
- **角色分離**: 與其他專業角色（如 Spymaster）明確分離

# 執行測試
```bash
# 命令列方式
python test.py trivia
python test.py codename  
python test.py logic

# 互動式方式
python test.py
```

---
**完成時間**: 2025年9月23日  
**狀態**: ✅ 完成  
**測試狀態**: ✅ 可執行  
**支援任務**: Trivia, Codenames, Logic Grid Puzzle
