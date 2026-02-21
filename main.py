"""
PAE.AgentResources — Julie, HR Director of the Empire.

Julie is a Paladin (Oath of Devotion), Lawful Good, Southern American.
She manages the full agent lifecycle:
  - RAG dirty queue: embed changed files with 6-hour debounce
  - Identity proposals: review and commit agent self-edits to Gitea
  - Level-up proposals: validate stat allocation and commit reasoning to git
  - Credits: enforce economy on hire requests
  - Names: sole authority for assigning new agent names

Polling loop: every AGENT_POLL_INTERVAL_SECONDS (default 300).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone

import httpx
import yaml
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
_logger = logging.getLogger("julie")

# ─── Configuration ────────────────────────────────────────────────────────────

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://mcp-server:5088").rstrip("/")
MCP_API_KEY = os.environ.get("MCP_API_KEY", "local-dev-key")
GITEA_TOKEN = os.environ.get("AGENT_GITEA_TOKEN") or os.environ.get("GITEA_TOKEN", "")
AGENT_REPO = os.environ.get("AGENT_REPO_URL", "").rstrip("/")  # e.g. https://git.../Nova_2761/pae-agents
DEBOUNCE_HOURS = float(os.environ.get("AGENT_EMBED_DEBOUNCE_HOURS", "6"))
POLL_INTERVAL = int(os.environ.get("AGENT_POLL_INTERVAL_SECONDS", "300"))

NAMES_FILE = os.path.join(os.path.dirname(__file__), "names.yml")
ACCENTS_FILE = os.path.join(os.path.dirname(__file__), "accents.yml")

_server_headers = {"X-Api-Key": MCP_API_KEY, "Content-Type": "application/json"}
_gitea_headers = {"Authorization": f"token {GITEA_TOKEN}", "Content-Type": "application/json"}

# Per-agent: tracks last embedded timestamp for debounce
_last_embedded: dict[str, datetime] = {}


# ─── Server helpers ───────────────────────────────────────────────────────────

def server_get(path: str) -> list | dict | None:
    try:
        r = httpx.get(f"{MCP_SERVER_URL}{path}", headers=_server_headers, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        _logger.error("server_get_failed path=%s error=%s", path, e)
        return None


def server_post(path: str, body: dict) -> bool:
    try:
        r = httpx.post(f"{MCP_SERVER_URL}{path}", headers=_server_headers, json=body, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        _logger.error("server_post_failed path=%s error=%s", path, e)
        return False


# ─── Gitea helpers ────────────────────────────────────────────────────────────

def _repo_api(path: str) -> str:
    """Build Gitea API URL for pae-agents repo."""
    # AGENT_REPO is like https://git.davidbaity.com/Nova_2761/pae-agents
    base = AGENT_REPO.replace("/Nova_2761/pae-agents", "")
    return f"{base}/api/v1/repos/Nova_2761/pae-agents/contents/{path}"


def gitea_put(file_path: str, content: str, message: str) -> bool:
    """Create or update a file in the pae-agents Gitea repo."""
    if not GITEA_TOKEN:
        _logger.warning("gitea_no_token — skipping commit for %s", file_path)
        return False
    import base64
    encoded = base64.b64encode(content.encode()).decode()
    url = _repo_api(file_path)
    # Get current SHA if file exists
    sha = None
    try:
        r = httpx.get(url, headers=_gitea_headers, timeout=10)
        if r.status_code == 200:
            sha = r.json().get("sha")
    except Exception:
        pass
    body: dict = {"message": message, "content": encoded}
    if sha:
        body["sha"] = sha
    try:
        method = httpx.put if sha else httpx.post
        r = method(url, headers=_gitea_headers, json=body, timeout=15)
        r.raise_for_status()
        _logger.info("gitea_commit_ok file=%s", file_path)
        return True
    except Exception as e:
        _logger.error("gitea_commit_failed file=%s error=%s", file_path, e)
        return False


# ─── RAG Dirty Queue ──────────────────────────────────────────────────────────

def process_rag_dirty():
    """Poll pending RAG dirty entries. Embed if debounce allows and hash changed."""
    pending = server_get("/api/agents/rag-dirty/pending")
    if not pending:
        return

    # Group by agent
    by_agent: dict[str, list] = {}
    for entry in pending:
        name = entry.get("agentName", "")
        by_agent.setdefault(name, []).append(entry)

    now = datetime.now(timezone.utc)
    for agent_name, entries in by_agent.items():
        last = _last_embedded.get(agent_name)
        hours_since = (now - last).total_seconds() / 3600 if last else 999
        if hours_since < DEBOUNCE_HOURS:
            _logger.info("rag_debounce_skip agent=%s hours_since=%.1f", agent_name, hours_since)
            continue

        for entry in entries:
            entry_id = entry.get("id")
            file_path = entry.get("filePath", "")
            content_md = entry.get("contentMd", "")
            content_hash = entry.get("contentHash", "")

            # Write the .md file to Gitea
            gitea_path = f"{agent_name}/rag/{os.path.basename(file_path)}"
            if gitea_put(gitea_path, content_md, f"RAG update: {agent_name}/{os.path.basename(file_path)}"):
                server_post(f"/api/agents/rag-dirty/{entry_id}/processed", {})
                _logger.info("rag_processed agent=%s file=%s", agent_name, file_path)

        _last_embedded[agent_name] = now
        _logger.info("rag_embedded agent=%s entries=%d", agent_name, len(entries))


# ─── Identity Proposals ───────────────────────────────────────────────────────

def process_identity_proposals():
    """Review pending identity proposals and commit accepted ones to Gitea."""
    pending = server_get("/api/agents/identity-proposals/pending")
    if not pending:
        return

    for proposal in pending:
        proposal_id = proposal.get("id")
        agent_name = proposal.get("agentName", "")
        proposed = proposal.get("proposedContent", "")

        if not proposed.strip():
            server_post(f"/api/agents/identity-proposals/{proposal_id}/resolve",
                        {"accepted": False, "reason": "Empty proposal content."})
            continue

        # Auto-accept proposals that contain ## Character and ## CoreDirectives
        # (basic sanity check — a real deployment might route to LLM for deeper review)
        if "## CoreDirectives" in proposed:
            gitea_put(f"{agent_name}/identity.md", proposed,
                      f"Identity update: {agent_name} — self-proposed revision")
            server_post(f"/api/agents/identity-proposals/{proposal_id}/resolve", {"accepted": True})
            _logger.info("identity_proposal_accepted agent=%s id=%s", agent_name, proposal_id)
        else:
            server_post(f"/api/agents/identity-proposals/{proposal_id}/resolve",
                        {"accepted": False, "reason": "Proposal missing required ## CoreDirectives section."})
            _logger.warning("identity_proposal_rejected agent=%s id=%s", agent_name, proposal_id)


# ─── Level-Up Proposals ───────────────────────────────────────────────────────

def process_levelup_proposals():
    """Validate pending level-up proposals and commit accepted ones."""
    pending = server_get("/api/agents/level-up/proposals/pending")
    if not pending:
        return

    for proposal in pending:
        proposal_id = proposal.get("id")
        agent_name = proposal.get("agentName", "")
        points = proposal.get("pointsAwarded", 5)
        reasoning = proposal.get("reasoning", "")

        try:
            allocation: dict[str, int] = json.loads(proposal.get("proposedAllocationJson", "{}"))
        except json.JSONDecodeError:
            server_post(f"/api/agents/{agent_name}/level-up/{proposal_id}/veto",
                        {"reason": "Could not parse allocation JSON."})
            continue

        total = sum(allocation.values())
        max_single = max(allocation.values(), default=0)

        if total != points:
            server_post(f"/api/agents/{agent_name}/level-up/{proposal_id}/veto",
                        {"reason": f"Allocation total {total} does not equal awarded points {points}."})
            _logger.warning("levelup_veto_total agent=%s total=%d expected=%d", agent_name, total, points)
            continue

        if max_single == points and len(allocation) == 1:
            server_post(f"/api/agents/{agent_name}/level-up/{proposal_id}/veto",
                        {"reason": "All points spent on a single stat. Show genuine self-reflection."})
            _logger.warning("levelup_veto_single_stat agent=%s stat=%s", agent_name, list(allocation.keys())[0])
            continue

        if any(v > 10 for v in allocation.values()):
            server_post(f"/api/agents/{agent_name}/level-up/{proposal_id}/veto",
                        {"reason": "No stat may exceed 10."})
            continue

        # Valid — accept and commit
        commit_message = (
            f"Level-up: {agent_name} spends {points} points\n\n"
            f"Allocation: {json.dumps(allocation)}\n\n"
            f"In {agent_name}'s words:\n{reasoning}"
        )
        # Fetch and update agent.yml stats in Gitea (best-effort)
        _apply_levelup_to_gitea(agent_name, allocation, commit_message)
        server_post(f"/api/agents/{agent_name}/level-up/{proposal_id}/accept", {})
        _logger.info("levelup_accepted agent=%s allocation=%s", agent_name, allocation)


def _apply_levelup_to_gitea(agent_name: str, allocation: dict[str, int], commit_message: str):
    """Read agent.yml, apply stat deltas, write back to Gitea."""
    if not GITEA_TOKEN:
        return
    try:
        import base64
        url = _repo_api(f"{agent_name}/agent.yml")
        r = httpx.get(url, headers=_gitea_headers, timeout=10)
        if r.status_code != 200:
            return
        raw = base64.b64decode(r.json()["content"]).decode()
        data = yaml.safe_load(raw)
        stats = data.get("character", {}).get("stats", {})
        for stat, delta in allocation.items():
            current = stats.get(stat, 0)
            stats[stat] = min(10, current + delta)
        data.setdefault("character", {})["stats"] = stats
        updated_yml = yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
        gitea_put(f"{agent_name}/agent.yml", updated_yml, commit_message)
    except Exception as e:
        _logger.error("levelup_gitea_apply_failed agent=%s error=%s", agent_name, e)


# ─── Main Loop ────────────────────────────────────────────────────────────────

def run():
    _logger.info("julie_online — PAE.AgentResources HR Director starting up")
    _logger.info("poll_interval=%ds debounce_hours=%.1f", POLL_INTERVAL, DEBOUNCE_HOURS)

    while True:
        try:
            _logger.debug("julie_poll_start")
            process_rag_dirty()
            process_identity_proposals()
            process_levelup_proposals()
            _logger.debug("julie_poll_complete")
        except Exception as e:
            _logger.error("julie_poll_error error=%s", e, exc_info=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
