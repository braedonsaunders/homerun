"""
Skill discovery and loading.

Discovers SKILL.md files from three tiers (builtin, user, project),
parses YAML frontmatter for metadata, and provides lazy loading
of full skill instructions.

Inspired by virattt/dexter's three-tier skill system.
"""

import logging
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from utils.utcnow import utcnow
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    name: str
    description: str
    instructions: str  # Full markdown instructions
    tier: str  # "builtin", "user", "project"
    file_path: str
    # Optional metadata from frontmatter
    version: str = "1.0"
    author: str = ""
    tags: list[str] = field(default_factory=list)
    # For deduplication
    requires_tools: list[str] = field(default_factory=list)


class SkillLoader:
    """
    Discovers and loads skills from three tiers.

    Usage:
        loader = SkillLoader()
        loader.discover()  # Scan all tiers

        skills = loader.list_skills()
        skill = loader.get_skill("resolution_analysis")
        instructions = skill.instructions
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}  # name -> Skill
        self._discovered = False

        # Define tier directories
        self._builtin_dir = Path(__file__).parent / "builtin"
        self._user_dir = Path.home() / ".homerun" / "skills"
        self._project_dir = Path.cwd() / ".homerun" / "skills"

    def discover(self):
        """Scan all tiers for skill files. Later tiers override earlier."""
        self._skills.clear()

        # Tier 1: Builtin
        self._scan_directory(self._builtin_dir, "builtin")

        # Tier 2: User
        self._scan_directory(self._user_dir, "user")

        # Tier 3: Project
        self._scan_directory(self._project_dir, "project")

        self._discovered = True
        logger.info(
            f"Discovered {len(self._skills)} skills: {list(self._skills.keys())}"
        )

    def _scan_directory(self, directory: Path, tier: str):
        """Scan a directory for .md skill files."""
        if not directory.exists():
            return

        for md_file in directory.glob("*.md"):
            try:
                skill = self._parse_skill_file(md_file, tier)
                if skill:
                    self._skills[skill.name] = skill  # Later tiers override
            except Exception as e:
                logger.warning(f"Failed to parse skill file {md_file}: {e}")

    def _parse_skill_file(self, file_path: Path, tier: str) -> Optional[Skill]:
        """Parse a skill markdown file with YAML frontmatter."""
        content = file_path.read_text(encoding="utf-8")

        # Parse YAML frontmatter (between --- delimiters)
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1])
                instructions = parts[2].strip()
            else:
                return None
        else:
            return None

        if not frontmatter or "name" not in frontmatter:
            return None

        return Skill(
            name=frontmatter["name"],
            description=frontmatter.get("description", ""),
            instructions=instructions,
            tier=tier,
            file_path=str(file_path),
            version=frontmatter.get("version", "1.0"),
            author=frontmatter.get("author", ""),
            tags=frontmatter.get("tags", []),
            requires_tools=frontmatter.get("requires_tools", []),
        )

    def list_skills(self) -> list[dict]:
        """List all available skills with metadata (without full instructions)."""
        if not self._discovered:
            self.discover()

        return [
            {
                "name": s.name,
                "description": s.description,
                "tier": s.tier,
                "version": s.version,
                "tags": s.tags,
            }
            for s in self._skills.values()
        ]

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name. Returns None if not found."""
        if not self._discovered:
            self.discover()
        return self._skills.get(name)

    def get_skill_instructions(self, name: str) -> Optional[str]:
        """Get just the instructions for a skill. Lazy loading."""
        skill = self.get_skill(name)
        return skill.instructions if skill else None

    async def execute_skill(
        self,
        name: str,
        context: dict = None,
        model: str = None,
    ) -> dict:
        """
        Execute a skill by passing its instructions to an agent.

        The skill's instructions become the system prompt for an agent
        that follows the structured workflow defined in the skill.

        Returns the agent's result dict.
        """
        from models.database import AsyncSessionLocal, SkillExecution
        from services.ai.agent import run_agent_to_completion

        skill = self.get_skill(name)
        if not skill:
            raise ValueError(f"Skill not found: {name}")

        execution_id = uuid.uuid4().hex[:16]
        started_at = utcnow()

        # Record execution start
        async with AsyncSessionLocal() as session:
            execution = SkillExecution(
                id=execution_id,
                skill_name=name,
                input_context=context,
                status="running",
                started_at=started_at,
            )
            session.add(execution)
            await session.commit()

        try:
            # Build query from context
            query = (
                context.get("query", f"Execute the {name} analysis workflow.")
                if context
                else f"Execute the {name} analysis workflow."
            )

            # Run agent with skill instructions as system prompt
            result = await run_agent_to_completion(
                system_prompt=skill.instructions,
                query=query,
                model=model,
                max_iterations=10,
                session_type=f"skill_{name}",
                market_id=context.get("market_id") if context else None,
                opportunity_id=context.get("opportunity_id") if context else None,
            )

            # Update execution
            async with AsyncSessionLocal() as session:
                from sqlalchemy import select

                stmt = select(SkillExecution).where(SkillExecution.id == execution_id)
                row = (await session.execute(stmt)).scalar_one_or_none()
                if row:
                    row.status = "completed"
                    row.output_result = result
                    row.completed_at = utcnow()
                    row.duration_seconds = (
                        utcnow() - started_at
                    ).total_seconds()
                    if "session_id" in result:
                        row.session_id = result["session_id"]
                    await session.commit()

            return result

        except Exception as e:
            # Update execution with error
            async with AsyncSessionLocal() as session:
                from sqlalchemy import select

                stmt = select(SkillExecution).where(SkillExecution.id == execution_id)
                row = (await session.execute(stmt)).scalar_one_or_none()
                if row:
                    row.status = "failed"
                    row.error = str(e)
                    row.completed_at = utcnow()
                    row.duration_seconds = (
                        utcnow() - started_at
                    ).total_seconds()
                    await session.commit()
            raise


# Singleton
skill_loader = SkillLoader()
