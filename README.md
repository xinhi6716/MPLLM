# MPLLM-Nano: Multi-Persona Layered Language Model

**MPLLM-Nano** 是一個輕量級、模組化的多角色語言模型協作框架。
本專案基於 [Solo Performance Prompting (SPP)](https://arxiv.org/abs/2307.05300) 的概念進行了深度重構，引入了 **依賴注入 (Dependency Injection)** 與 **分層架構 (Layered Architecture)**，旨在解決複雜推理任務（如邏輯謎題、創意寫作、聯想猜謎）。

---

##  核心特色 (Key Features)

* **極致模組化 (Modularity)**：將 LLM 推理流程拆解為 `Switch` (分派)、`Researcher` (搜集)、`Thinker` (推理)、`Decider` (決策) 等獨立層級。
* **依賴注入 (Nano Engine)**：核心 `nano()` 函式與模型解耦，可輕易替換底層模型（如 GPT-4o, Llama 3, Azure OpenAI）。
* **魯棒性設計 (Robustness)**：內建 Switch Layer 的 **Fallback 機制**，當模型生成的 JSON 格式錯誤或為空時，自動注入預設專家角色，確保流程不中斷。
* **自動化評測 (Auto-Evaluation)**：
    * 內建針對 **Trivia** (關鍵字覆蓋率)、**Codenames** (目標詞命中)、**Logic Puzzle** (選項匹配) 的評分器。
    * 支援 **Regex 智慧抓取**，即使模型輸出包含冗長推論，也能精準抓取 `Final Answer: X` 格式。
* **完整數據追蹤**：自動生成 CSV 報表，記錄每題的 Token 消耗、預估成本 (USD)、執行時間與詳細評分結果。

---

##  目前進度與已知限制 (Status & Limitations)

### ✅ 已完成功能 (Implemented)
1.  **全架構重構**：完成從 SPP 到 MPLLM-Nano 的遷移，建立 `core`, `layers`, `utils` 分層結構。
2.  **三大任務支援**：
    * **Trivia Creative Writing**：Researcher 負責查證，Decider 負責寫作。
    * **Codenames**：Spymaster 角色模擬與目標詞聯想。
    * **Logic Grid Puzzle**：多角色演繹推理與邏輯仲裁。
3.  **評分系統修復**：Logic Puzzle 評分器已升級為 Regex 模式，解決了因 Decider 解釋過多而被誤判為錯誤的問題。
4.  **日誌系統增強**：解決了 Windows 下 Excel 檔案鎖定導致的 `Permission Denied` 崩潰問題；支援自動建立 `test_DATA` 目錄。

### ⚠️ 已知限制與待優化 (Known Issues & Future Work)
1.  **模型推理能力瓶頸 (Model Capability)**：
    * 在 Logic Puzzle 任務中，雖然程式邏輯正確（Pipeline 跑通、評分器抓到答案），但 **GPT-4o-mini** 的推理準確率仍偏低（例如將選項 2 誤判為 1）。
    * *建議方案*：對於複雜邏輯任務，需考慮切換至更強的模型（如 GPT-4o）或進一步優化 CoT (Chain-of-Thought) 提示詞。
2.  **Switch Layer 穩定性**：
    * 小型模型偶爾會無法輸出符合規範的 JSON 格式，導致觸發 Fallback 機制使用預設 Persona。
    * *建議方案*：雖然已有救援機制，但長期應微調 Switch Layer 的 Prompt，增加 Few-Shot 範例以提升 JSON 遵循率。
3.  **API 依賴性**：
    * 目前 `utils/api_client.py` 主要針對 OpenAI API 介面實作。若需串接 HuggingFace 本地模型需自行擴充適配器。

---

## 📂 Project Structure

```text
MPLLM/
├── core/                   # 核心引擎
│   ├── nano.py             # 依賴注入核心 (Dependency Injection)
│   └── tracker.py          # Token 計數與成本計算器
├── layers/                 # 角色層級邏輯 (Persona Layers)
│   ├── switch.py           # Layer 1: 任務分派與角色生成 (含 Fallback)
│   ├── researcher.py       # Layer 2: 資訊檢索與背景補充
│   ├── thinker.py          # Layer 3: 多路並行推理 (Parallel Reasoning)
│   ├── minimux.py          # Layer 4: 資訊整合 (Multiplexer)
│   └── guesser.py          # Layer 5: 最終輸出生成
├── prompts/                # 提示詞庫
│   └── templates.py        # 集中管理各任務的 System Prompts
├── utils/                  # 工具組
│   ├── api_client.py       # OpenAI API 客戶端封裝
│   ├── data_loader.py      # 支援 .jsonl 格式的資料讀取器 (含答案解析)
│   ├── evaluator.py        # 自動評分邏輯 (Trivia, Codenames, Logic Regex)
│   └── logger.py           # CSV 報表生成器 (自動處理檔案鎖定)
├── data/                   # 測試數據集
│   ├── trivia/             # Trivia Creative Writing
│   ├── codenames/          # Codenames
│   └── logic/              # Logic Grid Puzzle
├── pipeline_core.py        # 主流水線邏輯 (串接各層與格式化輸入)
└── main.py                 # 程式入口 (CLI 介面)
```
##  Installation

1.  **建立虛擬環境 (推薦)**
    ```bash
    python -m venv venv
    # Windows:
    source venv/Scripts/activate
    # Mac/Linux:
    source venv/bin/activate
    ```

2.  **安裝依賴**
    ```bash
    pip install -r requirements.txt
    ```

3.  **設定 API Key**
    請確保環境變數中包含 `OPENAI_API_KEY`。
    ```bash
    # Windows PowerShell:
    $env:OPENAI_API_KEY="sk-..."
    
    # Mac/Linux:
    export OPENAI_API_KEY="sk-..."
    ```

---

## 🏃 使用方法 (Usage)

透過 `main.py` 執行測試。支援單題測試 (Debugging) 與批量測試 (Batch Experiment)。

### 1. Logic Grid Puzzle (邏輯推理)
測試模型的多步推理能力與排除法。
```bash
# 測試單題 (Limit 1) - 用於確認 Pipeline 是否跑通
python main.py --task logic --data data/logic_grid_puzzle/logic_grid_puzzle_200.jsonl --limit 1

# 批量測試 (前 50 題) - 用於收集統計數據
python main.py --task logic --data data/logic_grid_puzzle/logic_grid_puzzle_200.jsonl --limit 50
```
### 2. Trivia Creative Writing (創意寫作)
測試模型的知識檢索與整合寫作能力。
```bash
python main.py --task trivia --data data/trivia_creative_writing/trivia_creative_writing_100_n_5.jsonl --limit 5
```
### 3. Codenames (聯想猜謎)
測試 Spymaster 角色的語義聯想能力。
```bash
python main.py --task codenames --data data/codenames_collaborative/codenames_50.jsonl
```
## 📊 輸出結果範例 (Outputs)

每次執行結束後，系統會自動在 `test_DATA/` 資料夾中生成 CSV 報表。

* **檔名格式**：`MPLLM_{Model}_{Mode}_{Date}.csv`
* **內容範例**：

| Run_ID | Task_Type | Final_Answer | Score | Eval_Details | Execution_Time_Sec |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | logic | Final Answer: 2 | 1.0 | Correct (Regex) | 27.80 |
| 2 | trivia | The story begins... | 0.8 | 4/5 Correct | 45.26 |

---
*Based on the work: "Unleashing Cognitive Synergy in Large Language Models: A Task-Solving Agent through Multi-Persona Self-Collaboration" (Wang et al., 2023)*
