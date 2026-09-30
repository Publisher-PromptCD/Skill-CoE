"""Skill writing examples adapted from official SkillOpt artifacts."""

SKILL_STYLE = """A Skill is reusable operational guidance for the acting agent.
Write an applicability condition followed by concrete actions in useful order,
retaining supported prerequisites, exceptions and completion/check conditions.
Use a short procedure for dependent steps, or a specific warning for an observed
pitfall. Avoid vague advice such as 'be careful' or 'verify everything' when the
evidence supplies an executable procedure. Keep episode narration and speculation
out of the rule; put evidence IDs in their separate field when the schema allows it.

Style demonstrations adapted from SkillOpt, NOT current-task evidence:
- When moving multiple objects with a one-object carrying limit, choose one target
  receptacle, deliver the first object, then fetch and deliver the next to that same
  target. Leave delivered objects there; use a remembered source for the next pickup.
- When a question asks for an author but a search result names a book, use the book
  as context and extract its author from the supporting passage; do not return the
  book title as the answer.
These demonstrate specificity and scope, not knowledge to copy into this library.
Generate content only from supplied task evidence. Never cite these demonstrations
as observations or add their procedures without independent task support.
"""
