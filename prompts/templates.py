# prompts/templates.py

class MPLLMPrompts:
    # --- 通用設定 ---
    DEFAULT_SYSTEM = "You are a helpful AI assistant."

    # --- Task 1: Trivia Creative Writing ---
    TRIVIA_SWITCH_SYSTEM = """You are a Persona Generator.
Given a topic, create 3 distinct groups of experts (Researcher + Thinker) to collaborate on writing a story.
Format: JSON object with specific keys."""
    
    TRIVIA_SWITCH_USER = """Topic: {topic}
Output JSON format:
{{
  "groups": [
    {{"id": 1, "researcher": "...", "thinker": "..."}},
    {{"id": 2, "researcher": "...", "thinker": "..."}},
    {{"id": 3, "researcher": "...", "thinker": "..."}}
  ],
  "decider": "..."
}}"""

    TRIVIA_WRITER = """Write a short, coherent story about {topic} that incorporates the answers to the following {n} questions: {questions}.
Base your story on the best ideas provided."""

    # --- Task 2: Codenames ---
    CODENAMES_SWITCH_SYSTEM = """You are Personas Switch for a Codenames-style task."""
    
    CODENAMES_SWITCH_USER = """== Inputs ==
- n = {n}
- target_words = {target_words}
- word_list = {word_list}

== Your goals ==
1) As Spymaster, output ONE English word clue.
2) Select three diverse persona groups (researcher + thinker).
3) Define ONE global decider.

Output STRICT JSON:
{{
  "spymaster_clue": "word",
  "groups": [
    {{"group_id": 1, "researcher": {{ "persona": "..." }}, "thinker": {{ "persona": "..." }} }},
    {{"group_id": 2, "researcher": {{ "persona": "..." }}, "thinker": {{ "persona": "..." }} }},
    {{"group_id": 3, "researcher": {{ "persona": "..." }}, "thinker": {{ "persona": "..." }} }}
  ],
  "decider": {{ "persona": "..." }}
}}"""

    # --- Task 3: Logic Grid Puzzle (修復版) ---
    # 這裡恢復了詳細的規則與 JSON 範例，確保模型能正確生成 groups 雖然我不知道為什麼要這麼做
    LOGIC_SWITCH_SYSTEM = """You are Personas Switch for a Logic Grid Puzzle.
Your goal is to breakdown the puzzle and assign roles to solve it."""
    
    LOGIC_SWITCH_USER = """== Input ==
{inputs}

== Your goals ==
1) Select three complementary persona groups:
   - researcher: extracts constraints (e.g., 'left of', 'next to').
   - thinker: performs step-by-step deduction.
2) Define one global decider persona who integrates the findings.

== Output schema (STRICT JSON ONLY) ==
{{
  "groups": [
    {{"group_id": 1, "researcher": {{"persona": "..."}}, "thinker": {{"persona": "..."}} }},
    {{"group_id": 2, "researcher": {{"persona": "..."}}, "thinker": {{"persona": "..."}} }},
    {{"group_id": 3, "researcher": {{"persona": "..."}}, "thinker": {{"persona": "..."}} }}
  ],
  "decider": {{"persona": "..."}}
}}
IMPORTANT: Return ONLY the JSON object. No markdown formatting."""