MPLLM-Nano: Multi-Persona Layered Language Model
MPLLM-Nano 是一個輕量級、模組化的多角色語言模型協作框架。 本專案基於 Solo Performance Prompting (SPP) 的概念進行了深度重構，引入了 依賴注入 (Dependency Injection) 與 分層架構 (Layered Architecture)，旨在解決複雜推理任務（如邏輯謎題、創意寫作、聯想猜謎）。

🚀 核心特色 (Key Features)
極致模組化 (Modularity)：將 LLM 推理流程拆解為 Switch (分派)、Researcher (搜集)、Thinker (推理)、Decider (決策) 等獨立層級。

依賴注入 (Nano Engine)：核心 nano() 函式與模型解耦，可輕易替換底層模型（如 GPT-4o, Llama 3, Azure OpenAI）。

魯棒性設計 (Robustness)：內建 Fallback 機制，當模型生成的 JSON 格式錯誤時，自動注入預設 Persona 確保流程不中斷。

自動化評測 (Auto-Evaluation)：

內建針對 Trivia (關鍵字覆蓋率)、Codenames (目標詞命中)、Logic Puzzle (選項匹配) 的評分器。

支援 Regex 智慧抓取，即使模型廢話連篇也能精準抓到 Final Answer。

完整數據追蹤：自動生成 CSV 報表，記錄每題的 Token 消耗、預估成本 (USD) 與執行時間。

📂 專案結構 (Project Structure)
Plaintext

MPLLM/
├── core/                   # 核心引擎
│   ├── nano.py             # 依賴注入核心 (Dependency Injection)
│   └── tracker.py          # Token 計數與成本計算器
├── layers/                 # 角色層級邏輯 (Persona Layers)
│   ├── switch.py           # Layer 1: 任務分派與角色生成
│   ├── researcher.py       # Layer 2: 資訊檢索與背景補充
│   ├── thinker.py          # Layer 3: 多路並行推理 (Parallel Reasoning)
│   ├── minimux.py          # Layer 4: 資訊整合 (Multiplexer)
│   └── guesser.py          # Layer 5: 最終輸出生成
├── prompts/                # 提示詞庫
│   └── templates.py        # 集中管理各任務的 System Prompts (Trivia, Codenames, Logic)
├── utils/                  # 工具組
│   ├── api_client.py       # OpenAI API 客戶端封裝
│   ├── data_loader.py      # 支援 .jsonl 格式的資料讀取器
│   ├── evaluator.py        # 自動評分邏輯 (含 Regex 答案提取)
│   └── logger.py           # CSV 報表生成器 (自動處理檔案鎖定與路徑)
├── data/                   # 測試數據集
│   ├── trivia/             # Trivia Creative Writing 數據
│   ├── codenames/          # Codenames 數據
│   └── logic/              # Logic Grid Puzzle 數據
├── pipeline_core.py        # 主流水線邏輯 (串接各層)
└── main.py                 # 程式入口 (CLI 介面)
🛠️ 安裝與設定 (Installation)
建立虛擬環境 (推薦)

Bash

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
安裝依賴

Bash

pip install -r requirements.txt
(主要依賴為 openai)

設定 API Key 請確保環境變數中包含 OPENAI_API_KEY，或直接修改 utils/api_client.py。

Bash

export OPENAI_API_KEY="sk-..."
# Windows PowerShell:
# $env:OPENAI_API_KEY="sk-..."
🏃 使用方法 (Usage)
透過 main.py 執行測試。支援單題測試 (Debugging) 與批量測試 (Batch Experiment)。

1. Logic Grid Puzzle (邏輯推理)
測試模型的多步推理能力與排除法。

Bash

# 測試單題 (Limit 1)
python main.py --task logic --data data/logic_grid_puzzle/logic_grid_puzzle_200.jsonl --limit 1

# 批量測試 (前 50 題)
python main.py --task logic --data data/logic_grid_puzzle/logic_grid_puzzle_200.jsonl --limit 50
2. Trivia Creative Writing (創意寫作)
測試模型的知識檢索與整合寫作能力。

Bash

python main.py --task trivia --data data/trivia_creative_writing/trivia_creative_writing_100_n_5.jsonl --limit 5
3. Codenames (聯想猜謎)
測試 Spymaster 角色的語義聯想能力。

Bash

python main.py --task codenames --data data/codenames_collaborative/codenames_50.jsonl --limit 5
📊 輸出結果 (Outputs)
每次執行結束後，系統會自動在 test_DATA/ 資料夾中生成 CSV 報表。

檔名格式：MPLLM_{Model}_{Mode}_{Date}.csv

欄位說明：

Score: 該題得分 (0.0 ~ 1.0)。

Eval_Details: 評分詳情 (例如 Correct (Regex) 或 Wrong (Pred: 1))。

Execution_Time_Sec: 執行秒數。

Cost_USD: 該次執行的預估 API 成本。

📝 開發日誌 (Changelog)
v1.0 - 2025-09-23 (Current)
✅ Refactor: 完成從 SPP 到 MPLLM-Nano 的全架構重構。

✅ Fix: 修復 Logic Puzzle 評分器，加入 Regex 支援以解決 Decider 輸出格式導致的誤判 (0分問題)。

✅ Feature: 新增 Switch Layer 的 Fallback 機制，當模型 JSON 解析失敗時自動注入預設專家角色。

✅ Feature: 實作 utils/logger.py，支援防止檔案鎖定 (Permission Denied) 與自動建立目錄。

Based on the work: "Unleashing Cognitive Synergy in Large Language Models: A Task-Solving Agent through Multi-Persona Self-Collaboration" (Wang et al., 2023)
