# prompts/templates.py

class MPLLMPrompts:
    # --- 通用設定 ---
    DEFAULT_SYSTEM = "You are a helpful AI assistant."

    # ==========================================
    # TASK 1: Trivia Creative Writing
    # ==========================================
    
    # [Switch Layer] 負責生成角色
    TRIVIA_SWITCH_SYSTEM = """You are a Persona Generator.
Given a topic, create 3 distinct groups of experts (Researcher + Thinker) to collaborate on writing a story.
Format: JSON object with specific keys."""

    # 來自你的 switch_prompts.py -> trivia_personas_switch_prompt
    TRIVIA_SWITCH_USER = """topic: {topic}
Create 3 groups (roles: r=researcher, t=thinker).

Use 1-2 word topic related job titles.
- Researcher: topic expert who answers questions
- Thinker: topic writer who creates stories  
- no extra text; no explanation. just output JSON.

Rules: all 6 titles unique:
{{
"groups":[
  {{"g":1,"r":"","t":""}},
  {{"g":2,"r":"","t":""}},
  {{"g":3,"r":"","t":""}}
]}}"""

    # [Researcher Layer] 來自你的 prompts.py -> trivia_researcher_prompt
    TRIVIA_RESEARCHER = """You are a Multi-Perspective Research Engine.
Task: Answer trivia questions from three DISTINCT viewpoints sequentially.

[Global Context]
Questions: {questions}
Constraints: One word/phrase per answer. No explanations.

[Process]
Step 1: Adopt Persona 1 ({r1}). Scan questions. Output answers.
Step 2: Reset. Adopt Persona 2 ({r2}). Scan questions. Output answers.
Step 3: Reset. Adopt Persona 3 ({r3}). Scan questions. Output answers.

[Isolation Rules]
- When acting as R2, do NOT copy R1. Use your unique background ({r2}).
- When acting as R3, do NOT copy R1 or R2. Use your unique background ({r3}).
- If a question is outside your specific persona's knowledge, output "unknown".

[Output Format]
Strict JSON only. No markdown. No reasoning text.
{{
"r1": ["ans1_q1", "ans1_q2", ...],
"r2": ["ans2_q1", "ans2_q2", ...],
"r3": ["ans3_q1", "ans3_q2", ...]
}}"""

    # [Thinker Layer] 來自你的 prompts.py -> trivia_thinker_prompt
    TRIVIA_THINKER = """{thinker_persona} for {topic}:
Brief story approach (10 words max).
no extra text; no explanation.
Output JSON only: {{"creative_direction":"..."}}"""

    # [Minimux Layer] 來自你的 prompts.py -> trivia_minimux_prompt
    TRIVIA_MINIMUX = """You are the final judge for trivia creative writing.

Topic: {topic}
Questions: {questions}
Answers: R1={researcher1}, R2={researcher2}, R3={researcher3}
Directions: T1={thinker1}, T2={thinker2}, T3={thinker3}
Don't explain your reasoning, just output JSON.

Task 1: Select the BEST answer for each question
For each question Q[i], execute the following logic:
1. **Consensus Check (Coherence):** Prioritize answers agreed upon by 2 or 3 Researchers.
2. **Quality Arbitrage (Criticality):** If answers conflict, or if any answer is 'unknown' with a reason, utilize your Mini-Model expertise to critically evaluate the provided answers. Your final choice must be the most factually accurate, even if it deviates from the simple majority.
3. **Final Answer Constraint:** Ensure each output answer is ONE word or short phrase. You MUST NOT output 'unknown' as a final answer.

Task 2: Write integrated story
CRITICAL: The story MUST be about the topic "{topic}". Write ONE cohesive story (40-60 words) that incorporates ALL final answers naturally. Follow the creative directions T1, T2, T3.

Output JSON only (ASCII, double quotes):
{{"final_answers":["answer1","answer2","answer3","answer4","answer5"], "final_story":"..."}}"""

    # ==========================================
    # TASK 2: Codenames
    # ==========================================

    # [Switch Layer] 來自你的 switch_prompts.py -> codenames_personas_switch_prompt
    CODENAMES_SWITCH_SYSTEM = """You are Personas Switch for a Codenames-style task."""
    
    CODENAMES_SWITCH_USER = """Targets: {target_words}

Task:
- Pick ONE English clue word linking all targets.
- Create 3 groups. Each group has:
  - g: group number (1, 2, or 3)
  - r: researcher persona name (full name)
  - t: thinker persona name (full name)

Rules:
- Clue is ONE word.
- All 6 persona names must be different.
- Group numbers must be 1, 2, 3.
- No explanations. No extra text.

Return JSON only with keys:
"initial_clue" and "groups".
"""

    # [Researcher Layer] 來自你的 prompts.py -> codenames_researcher_prompt
    CODENAMES_RESEARCHER = """Three evaluators: {r1_persona}, {r2_persona}, {r3_persona}.
Each one thinks separately.

Clue: "{clue}"
Targets: {target_words}
Others: {others}

Task:
For each evaluator, say if the clue is closer to Targets or Others.
risk = high (others closer), medium (equal), low (targets closer)
coverage = number of targets that seem close (0-10)

Output JSON only with keys "r1", "r2", "r3".
Each key maps to an object with "risk" (string: low/medium/high) and "coverage" (number 0-10)."""

    # [Thinker Layer] 來自你的 prompts.py -> codenames_thinker_prompt
    CODENAMES_THINKER = """You are {thinker_persona}.

Clue: "{clue}"
Targets: {target_words}

Give one advice (max 10 words) for the clue. No extra text.

Output JSON only. Use one key "advice" whose value is your short advice string."""

    # [Minimux Layer] 來自你的 prompts.py -> codenames_minimux_prompt
    CODENAMES_MINIMUX = """You are Minimux, the final decision-maker for Codenames.

Target words: {target_words}
Word list: {word_list}
Original clue: "{clue_switch}"

Researcher results:
{researcher_result}

Thinker advice:
{thinker_result}

Task:
Decide the final clue using ONLY the researcher risks, researcher coverage,
and thinker short advice. No deep reasoning. No explanation.

Rules for keeping the original clue:
- Majority of researcher risks are "low"
- AND average researcher coverage >= 5
- AND thinker advice has no strongly negative signals

If these conditions are met:
- final_clue = original clue

Otherwise:
- Generate ONE NEW English clue word:
  * Must link ALL target words
  * Must NOT be in word_list
  * Must be ONE word only
  * Should be safer and clearer than the original

Output JSON only:
{{
  "final_clue": "..."
}}"""

    # [Guesser Layer] 來自你的 prompts.py -> codenames_guesser_prompt
    CODENAMES_GUESSER = """You are a Guesser in Codenames game.

Task: Based on the clue, select {k} words from the board that are most likely related to the clue.

Clue: {clue}
Board (25 words): {board}
Target word count: {k}

Instructions:
1. Analyze the semantic relationship between the clue and each word on the board.
2. Select exactly {k} words that are most strongly associated with the clue.
3. Consider semantic similarity, common associations, and thematic connections.
4. Output only a comma-separated list of {k} words.

Output format (comma-separated list only):
word1, word2, word3, word4
"""

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