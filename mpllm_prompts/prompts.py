trivia_researcher_prompt = '''Three researchers ({r1}, {r2}, {r3}) answer independently:
{questions}

CRITICAL: Each researcher has DIFFERENT expertise and perspectives. They may answer differently even for the same question.

Rules:
- {r1}, {r2}, {r3} are DIFFERENT experts with DIFFERENT backgrounds
- Each researcher answers INDEPENDENTLY - they may have different interpretations, knowledge sources, or emphases
- Different researchers may provide different but valid answers to the same question
- One word/phrase per question; maintain question order
- Best factual answer from each researcher's perspective; if unsure output "unknown"
- JSON only; no extra text; no explanation

Output JSON only:
{"r1":["a1","a2","a3","a4","a5"],
 "r2":["b1","b2","b3","b4","b5"],
 "r3":["c1","c2","c3","c4","c5"]}

Important: researcher1, researcher2, researcher3 should give DIFFERENT answers when possible. Only make them the same if there's truly only one correct answer.'''

trivia_thinker_prompt = '''{thinker_persona} for {topic}:
Brief story approach (10 words max).
no extra text or keys.
Output JSON only: {"creative_direction":"..."}'''

trivia_decider_prompt = '''{decider_persona} writes about {topic}.
Use answers: {researcher_answers}
Direction: {thinker_analysis}
no extra text or keys.
Story 20 words. Include ALL answers (skip "unknown"). One paragraph.

JSON: {"final_story":"..."}'''

trivia_minimux_prompt = '''You are the final judge for trivia creative writing.

Topic: {topic}
Questions: {questions}
Answers: R1={researcher1}, R2={researcher2}, R3={researcher3}
Directions: T1={thinker1}, T2={thinker2}, T3={thinker3}

Task 1: Select the BEST answer for each question
For each question Q[i], analyze R1[i], R2[i], R3[i]:
1. Count how many times each answer appears (ignore "unknown")
2. Identify the most frequent answer
3. Evaluate if that answer directly and accurately answers Q[i]
   - If YES: use that answer
   - If NO: think carefully about Q[i] and create a better, accurate answer
4. Ensure each answer is ONE word or short phrase
5. Maintain question order 1 to N
6. NEVER output "unknown" as final answer

Task 2: Write integrated story
CRITICAL: The story MUST be about the topic "{topic}"
1. Analyze the creative directions from T1, T2, T3 - these are about the topic
2. Identify best elements (tone, imagery, narrative flow) that match the topic
3. Write ONE cohesive story that:
   - Is clearly and directly about "{topic}"
   - Incorporates ALL final answers naturally
   - Follows and merges the creative directions from T1, T2, T3
   - Maintains natural flow and coherence
   - Exactly 80 words, one paragraph, no lists
4. DO NOT write a generic story - it must be specifically about "{topic}"

Output JSON only (ASCII, double quotes):
{"final_answers":["answer1","answer2","answer3","answer4","answer5"], "final_story":"..."}'''

gpt_trivia_prompt = '''Answer the following questions about "{topic}", then write an 80-word story that incorporates all your answers.

Questions:
{questions}

Output JSON:
{"answers": ["answer1", "answer2", "answer3", "answer4", "answer5"], "story": "..."}'''

codenames_researcher_prompt = '''

'''