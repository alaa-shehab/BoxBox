# Future ideas

Ideas that are out of scope for v1. Order and v2 scope are set in PLAN.md.

## v2 (agreed order)
1. `LiveFeed` via OpenF1. Real-time access is paid (sponsor tier), so it was out of v1 under the free-only rule.
2. Private prediction leagues with friends, plus a leaderboard.
3. Stewards' decision precedent search ("has this been penalized before?").
4. A 2025 → 2026 regulation diff explainer.
5. A post-race personalized debrief focused on the fan's favourites.

## Parking lot
- A NiceGUI or React frontend with true server push. Blocked on free hosting: Hugging Face Docker Spaces now require PRO.
- Alembic migrations once the schema grows beyond four tables.
- A semantic cache for repeated chat questions.
- A local small LLM (llama.cpp) as a last-resort fallback when free API quotas run out.
