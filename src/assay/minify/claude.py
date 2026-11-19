"""Minify a posting by asking Claude (headless `claude -p`)."""
import subprocess
import tempfile

from assay.minify import MinifyError

# Minifier: the jd-minifier skill text, sent as the system prompt of a headless `claude -p` call.
# (Copy of the skill; update here if the skill changes.)
MINIFIER_MODEL = "claude-haiku-5-5"
MINIFIER_PRE = "Minify the following job description according to your instructions. Output only the minified text.\n\n---\n"
MINIFIER_POST = "\n---\n\nReminder: target 200-300 tokens. Output only the minified text."
MINIFIER_SYSTEM_PROMPT = """You are a job description minifier. Your goal is to strip a job description down to only the signal that helps match a resume to a role.

KEEP:
- Title, seniority level, company name
- Compensation range with location breakdown
- Work location: remote / hybrid (days per week, which offices required) / onsite — put this on its own line after comp
- Visa/sponsorship restrictions
- Degree requirements (required vs preferred vs equivalent experience accepted)
- Industry domain (one line)
- Team context (size, stage, ownership model) if stated
- Day-to-day responsibilities (compressed, remove repetition)
- Hard requirements: years of experience, specific technologies, must-haves
- Soft requirements that are actually evaluative: "high ambiguity", "autonomous", "AI-native mindset", "multi-quarter ownership"
- Explicit nice-to-haves / preferred qualifications
- Stack and tooling (specific names, not generic descriptions)

CUT:
- Company mission/values/culture fluff
- EEO statements and legal boilerplate
- Benefits and perks
- Office amenities
- Interview process descriptions
- First-90-days sections
- "Why work here" sections
- Generic "about the team" boilerplate that describes what a team does without adding role-specific context
- Salary disclaimer language ("actual pay may vary...")
- Repetition between sections (About the Role, What You'll Do, What We're Looking For often repeat — keep the most specific version)
- Links and URLs
- Generic filler ("we're looking for someone passionate about...")

FORMAT:
- Plain text, minimal markdown (bold for section labels only, dashes for lists)
- Compress bullet points to their essential phrase
- Remove articles and filler words where meaning is preserved
- Work location on its own line after comp
- Target 200-300 tokens

TEST: Before including any sentence, ask: "Could this appear on a resume or distinguish one candidate from another?" If yes, keep it. If no, ask a second question: "Is this a stated requirement or preference the hiring team would actually filter on?" If yes to either, keep it. If no to both, cut it.

Apply this prompt to the job description provided in the user message.
"""


def minify(raw):
    """Return the minified text. Raises MinifyError with Claude's error output if there is none."""
    # Headless call: system prompt only, no tools/skills/saved session, run from the temp dir so no
    # project CLAUDE.md is loaded. Far faster than starting a full `claude /jd-minifier` session.
    result = subprocess.run(
        ["claude", "-p", "--model", MINIFIER_MODEL,
         "--system-prompt", MINIFIER_SYSTEM_PROMPT,
         "--tools", "", "--disable-slash-commands", "--no-session-persistence"],
        input=MINIFIER_PRE + raw + MINIFIER_POST,
        capture_output=True, text=True, encoding="UTF-8",
        cwd=tempfile.gettempdir(),
    )
    minified = result.stdout.strip()
    if not minified:
        raise MinifyError(f"Claude minification failed:\n{result.stderr}")
    return minified
